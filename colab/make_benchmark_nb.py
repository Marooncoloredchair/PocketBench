#!/usr/bin/env python3
r"""Rebuild `sbdd_robust_benchmark.ipynb` for DiffSBDD real100 (Colab). Run locally:

    python colab/make_benchmark_nb.py

"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [dedent(text).strip()]}


def py(text: str) -> dict:
    body = dedent(text).strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [body],
    }


def bundled_diffsbdd_real100_zlib_b64() -> str:
    """zlib-compress then base64-encode configs/diffsbdd_real100.yaml for Colab shim."""
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "diffsbdd_real100.yaml"
    blob = zlib.compress(yaml_path.read_bytes(), level=9)
    return base64.standard_b64encode(blob).decode("ascii")


_CELL6_CODE_TEMPLATE = '''
from pathlib import Path
import base64
import subprocess
import sys
import urllib.parse
import urllib.request
import zlib

ROOT = Path("/content/sbdd-robust").resolve()
SCRIPT = ROOT / "scripts" / "generate_real100_config.py"
CFG_REAL100 = ROOT / "configs" / "diffsbdd_real100.yaml"
RAW = ROOT / "data/raw/real100"


def download_pdbs_from_real100_yaml(cfg_path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML missing — rerun Cell 3") from exc

    with cfg_path.open(encoding="utf-8") as fh:
        yobj = yaml.safe_load(fh)
    pockets = yobj.get("pockets") if isinstance(yobj, dict) else None
    if not isinstance(pockets, list) or not pockets:
        raise RuntimeError(cfg_path.as_posix() + " has no pockets:[] — need generate_real100_config.py")

    RAW.mkdir(parents=True, exist_ok=True)
    for p in pockets:
        if not isinstance(p, dict):
            continue
        rel = Path(str(p.get("pdb") or "").strip()).name
        rel_low = rel.lower()
        if not rel_low.endswith(".pdb"):
            raise RuntimeError("invalid pdb path entry: " + str(p))
        pid = rel_low[:-4].strip().upper()
        if len(pid) != 4 or not pid.isalnum():
            raise RuntimeError("unexpected PDB ID from YAML: " + repr(pid))
        dest = RAW / rel_low
        if dest.is_file() and dest.stat().st_size > 500:
            continue
        url = "https://files.rcsb.org/download/" + urllib.parse.quote(pid) + ".pdb"
        urllib.request.urlretrieve(url, dest)


RAW.mkdir(parents=True, exist_ok=True)

if CFG_REAL100.is_file():
    # Repo ships the frozen 100-pocket panel; use it verbatim (the committed PDBs
    # already match) and only fetch any that are missing. Do NOT re-run the RCSB
    # generator here — it could select a different panel than the committed PDBs.
    print("Using committed configs/diffsbdd_real100.yaml (frozen panel); fetching any missing PDBs.")
    download_pdbs_from_real100_yaml(CFG_REAL100)
elif SCRIPT.is_file():
    print("$ python", SCRIPT)
    if subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT)).returncode != 0:
        raise RuntimeError("generator failed")
else:
    # Notebook embed (built by colab/make_benchmark_nb.py): clone may omit YAML on GitHub.
    _embedded = zlib.decompress(
        base64.standard_b64decode("@@DIFFSBDD_REAL100_ZLIB_B64@@")
    )
    CFG_REAL100.parent.mkdir(parents=True, exist_ok=True)
    CFG_REAL100.write_bytes(_embedded)
    print(
        "Wrote configs/diffsbdd_real100.yaml from notebook embed (frozen at notebook build)."
    )
    download_pdbs_from_real100_yaml(CFG_REAL100)

n_pdb = len(list(RAW.glob("*.pdb")))
print("PDB count:", n_pdb)
if n_pdb < 100:
    raise RuntimeError("<100 PDBs downloaded")
'''.strip(
    "\n"
)


CELL7 = dedent(r"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


DS = Path("/content/DiffSBDD")

# --- generate_ligands.py: torch.load + --device ---
gl = DS / "generate_ligands.py"
if gl.is_file():
    t = _read(gl)
    anchor = "\nfrom lightning_modules import LigandPocketDDPM\n"
    if anchor in t and "_torch_load_compat" not in t:
        shim = '''

_torch_load_orig = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _torch_load_orig(*args, weights_only=False, **kwargs)
        except TypeError:
            return _torch_load_orig(*args, **kwargs)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_compat
'''
        t = t.replace(anchor, shim + anchor, 1)

    if "parser.add_argument('--timesteps'" in t:
        needle = (
            "    parser.add_argument('--timesteps', type=int, default=None)\n"
            "    args = parser.parse_args()"
        )
        repl = (
            "    parser.add_argument('--timesteps', type=int, default=None)\n"
            "    parser.add_argument('--device', type=str, default=None)\n"
            "    args = parser.parse_args()"
        )
        if needle in t and "parser.add_argument('--device'" not in t:
            t = t.replace(needle, repl, 1)

    nd = (
        "    pdb_id = Path(args.pdbfile).stem\n\n"
        "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n\n"
        "    if args.batch_size is None:"
    )
    nd2 = (
        "    pdb_id = Path(args.pdbfile).stem\n\n"
        "    device = (\n"
        "        str(args.device).strip()\n"
        "        if args.device not in (None, '')\n"
        "        else ('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "    )\n\n"
        "    if args.batch_size is None:"
    )
    if nd in t:
        t = t.replace(nd, nd2, 1)

    _write(gl, t)
    print("generate_ligands.py patched OK")

LM = DS / "lightning_modules.py"
if LM.is_file():
    txt = _read(LM)
    lm_old = (
        "from Bio.PDB import PDBParser\n"
        "from Bio.PDB.Polypeptide import three_to_one\n"
    )
    lm_new = dedent('''
from Bio.PDB import PDBParser
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1

    def three_to_one(resname: str) -> str:
        r = (resname or '').strip().upper()
        if len(r) != 3:
            return 'X'
        amb = {'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'}
        if r in amb:
            return amb[r]
        try:
            return str(seq1(r))
        except Exception:
            return 'X'

''')
    if lm_old in txt:
        txt = txt.replace(lm_old, lm_new, 1)

    if "_diffsbdd_chain" not in txt:
        anchor = "from analysis.docking import smina_score\n\n\nclass LigandPocketDDPM"
        old_b = (
            "        if pocket_ids is not None:\n"
            "            # define pocket with list of residues\n"
            "            residues = [\n"
            "                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]\n"
            "                for x in pocket_ids]\n\n"
            "        else:"
        )
        new_b = (
            "        if pocket_ids is not None:\n"
            "            residues = []\n"
            "            for x in pocket_ids:\n"
            "                parts = str(x).split(':', 1)\n"
            "                if len(parts) != 2:\n"
            "                    raise ValueError(\n"
            '                        f"Bad pocket id {x!r}; expected chain:resseq")\n'
            "                chain = _diffsbdd_chain(pdb_struct, parts[0])\n"
            "                residues.append(_diffsbdd_residue(chain, int(parts[1])))\n\n"
            "        else:"
        )
        inserted = dedent('''
from analysis.docking import smina_score


def _diffsbdd_chain(pdb_model, chain_key: str):
    ck = (chain_key or '').strip()
    if ck in pdb_model:
        return pdb_model[ck]
    for cid in pdb_model:
        if str(cid).strip() == ck:
            return pdb_model[cid]
    raise KeyError(f"chain {chain_key!r} not in structure")


def _diffsbdd_residue(chain, resseq: int):
    rid0 = (" ", int(resseq), " ")
    if rid0 in chain:
        return chain[rid0]
    hits = [rid for rid in chain.child_dict if rid[1] == int(resseq)]
    if not hits:
        raise KeyError(f"resseq {resseq} not in chain {chain.id!r}")
    if len(hits) == 1:
        return chain[hits[0]]
    hits.sort(key=lambda r: (str(r[0]), str(r[2])))
    return chain[hits[0]]


class LigandPocketDDPM
''')
        if anchor in txt and old_b in txt:
            txt = txt.replace(anchor, inserted, 1).replace(old_b, new_b, 1)
            print("Inserted pocket residue helpers")
        else:
            print("WARN: residue block not matched upstream")

    _write(LM, txt)

pc_old_tpl = (
    "from Bio.PDB import PDBParser\n"
    "from Bio.PDB.Polypeptide import three_to_one, is_aa\n"
)
pc_new_tpl = dedent('''
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1

    def three_to_one(resname: str) -> str:
        r = (resname or '').strip().upper()
        if len(r) != 3:
            return 'X'
        amb = {'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'}
        if r in amb:
            return amb[r]
        try:
            return str(seq1(r))
        except Exception:
            return 'X'

''')
for fname in ("process_crossdock.py", "process_bindingmoad.py"):
    p = DS / fname
    if not p.is_file():
        continue
    t = _read(p)
    if pc_old_tpl in t:
        _write(p, t.replace(pc_old_tpl, pc_new_tpl, 1))

met = DS / "analysis" / "metrics.py"
if met.is_file():
    t = _read(met)
    if ":.3f} \\pm {" in t:
        _write(met, t.replace(":.3f} \\pm {", ":.3f} \\\\pm {", 1))

utils = DS / "utils.py"
if utils.is_file():
    txt = _read(utils)
    if "_colab_sddf_close_marker" not in txt:
        old_blk = '''def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    #with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m)

    # print(f'Wrote SDF file to {sdf_path}')
'''
        new_blk = '''def write_sdf_file(sdf_path, molecules):
    _colab_sddf_close_marker = True
    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    try:
        for m in molecules:
            if m is not None:
                w.write(m)
    finally:
        if hasattr(w, "close"):
            w.close()
        elif hasattr(w, "flush"):
            try:
                w.flush()
            except Exception:
                pass
'''
        if old_blk in txt:
            txt = txt.replace(old_blk, new_blk, 1)
        _write(utils, txt)

print("Cell 7 done")
""")


CELL8 = dedent(r"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML missing — reinstall Cell 3") from exc

ROOT = Path("/content/sbdd-robust")
src = ROOT / "configs" / "diffsbdd_real100.yaml"
if not src.is_file():
    raise RuntimeError(f"missing {src}")

with src.open(encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

# Checkpoint results+generations onto Google Drive when it is mounted (Cell 2) so a
# Colab disconnect mid-run is recoverable; fall back to ephemeral /content otherwise.
DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
if DRIVE_MYDRIVE.is_dir():
    WORK = DRIVE_MYDRIVE / "sbdd-robust-results" / "real100" / "work"
    on_drive = True
else:
    WORK = Path("/content/sbdd_robust_work/data")
    on_drive = False
RESULTS_DIR = WORK / "results"
GENERATIONS_DIR = WORK / "generations"

cfg.setdefault("paths", {})
cfg["project_root"] = str(ROOT)
cfg["paths"]["results"] = str(RESULTS_DIR)
cfg["paths"]["generations"] = str(GENERATIONS_DIR)
cfg["skip_failed_pockets"] = True
# Stable run_id + resume: re-running Cell 9 after a disconnect skips finished
# (pocket, perturbation) conditions instead of restarting from pocket 1.
cfg["run_id"] = "real100"
cfg["resume"] = True

m = cfg.setdefault("model", {})
m["repo_root"] = "/content/DiffSBDD"
m["checkpoint"] = "/content/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
m["python_exe"] = "/usr/bin/python3"
m.setdefault("sanitize", False)
m["extra_args"] = ["--device", "cuda"]

for d in (RESULTS_DIR, GENERATIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

outp = ROOT / "configs" / "diffsbdd_real100_colab.yaml"
with outp.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, width=140)

# Sidecar so later cells (9/10/11) use the same paths regardless of run order.
sidecar = ROOT / ".pb_paths.json"
sidecar.write_text(
    json.dumps(
        {
            "results": str(RESULTS_DIR),
            "generations": str(GENERATIONS_DIR),
            "run_id": "real100",
            "on_drive": on_drive,
        }
    ),
    encoding="utf-8",
)
print("wrote", outp)
print("results dir:", RESULTS_DIR, "(persisted to Drive)" if on_drive else "(EPHEMERAL /content)")
if not on_drive:
    print("WARNING: Drive not mounted (run Cell 2) — results will be LOST on disconnect.")
""")


CELL9 = dedent(r"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/sbdd-robust")
cfgp = ROOT / "configs" / "diffsbdd_real100_colab.yaml"
if not cfgp.is_file():
    raise RuntimeError("Run Cell 8 first.")

# Report where results are checkpointed (and whether a prior run can be resumed).
sidecar = ROOT / ".pb_paths.json"
if sidecar.is_file():
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    res_dir = Path(info["results"])
    prior = res_dir / f"metrics_per_condition__run{info.get('run_id', 'real100')}.csv"
    print("results dir:", res_dir, "(Drive)" if info.get("on_drive") else "(EPHEMERAL)")
    if prior.is_file():
        try:
            import pandas as pd

            ndone = len(pd.read_csv(prior))
            print(f"RESUME: found {ndone} completed conditions — these will be skipped.")
        except Exception:
            print("RESUME: prior checkpoint present; completed conditions will be skipped.")
    else:
        print("Fresh run (no prior checkpoint).")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["DIFFSBDD_PYTHON"] = "/usr/bin/python3"
os.environ["DIFFSBDD_REPO"] = str(Path("/content/DiffSBDD"))
os.environ["DIFFSBDD_CHECKPOINT"] = (
    "/content/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
)

cmd = [sys.executable, "-u", "-m", "sbdd_robust", "run", "--config", str(cfgp), "--resume"]
print("$ cd", ROOT)
print("$", " ".join(cmd))

proc = subprocess.Popen(
    cmd,
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

if proc.stdout is None:
    raise RuntimeError("subprocess.PIPE setup failed")

for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

rc = proc.wait()
if rc != 0:
    raise RuntimeError("sbdd_robust exited " + str(rc))
print("[benchmark finished] OK")
""")


CELL10 = dedent(r"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

_sidecar = Path("/content/sbdd-robust/.pb_paths.json")
if _sidecar.is_file():
    SRC = Path(json.loads(_sidecar.read_text(encoding="utf-8"))["results"])
else:
    SRC = Path("/content/sbdd_robust_work/data/results")
if not SRC.is_dir():
    raise RuntimeError("Missing results directory — Cell 9 must succeed first.")

dst = Path(DRIVE_OUT)


def mirror_tree(a: Path, b: Path) -> int:
    n = 0
    if b.is_dir():
        shutil.rmtree(b)
    b.mkdir(parents=True, exist_ok=True)
    for src in sorted(a.rglob("*")):
        rel = src.relative_to(a)
        target = b / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            n += 1
    return n


# Cell 8 checkpoints directly to Drive when it is mounted, so results may already
# live under dst — in that case just verify rather than copy onto itself.
src_res = SRC.resolve()
dst_res = dst.resolve()
already_on_drive = src_res == dst_res or str(src_res).startswith(str(dst_res) + "/")

if already_on_drive:
    print(f"Results already on Drive at {src_res} — no copy needed.")
    look_in = SRC
else:
    nfiles = mirror_tree(SRC, dst)
    print(f"Copied {nfiles} files into {dst_res}")
    look_in = dst

csvs = list(look_in.glob("metrics_per_condition__*.csv"))
if not csvs:
    raise RuntimeError("metrics_per_condition CSV not found")
print("Example metrics CSV:", csvs[0].name)
""")


CELL11 = dedent(r"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

_sidecar = Path("/content/sbdd-robust/.pb_paths.json")
if _sidecar.is_file():
    RES = Path(json.loads(_sidecar.read_text(encoding="utf-8"))["results"])
else:
    RES = Path("/content/sbdd_robust_work/data/results")
csvs = sorted(RES.glob("metrics_per_condition__*.csv"))
if not csvs:
    raise RuntimeError("No metrics_per_condition CSV")
metrics_csv = max(csvs, key=lambda p: p.stat().st_mtime)
print("Using:", metrics_csv)

import pandas as pd

df_raw = pd.read_csv(metrics_csv)
df = df_raw.copy()
for c in ("brittle_invariant", "brittleness_note"):
    if c in df.columns:
        df.drop(columns=c, inplace=True)

orig_rows = df[df["perturbation_tag"].astype(str) == "original"]
_nv = pd.to_numeric(orig_rows.get("n_valid", 0), errors="coerce").fillna(0)
coverage = float((_nv > 0).mean())
print(f"Coverage (original rows with n_valid>0, row-level fraction): {coverage:.5f}")

inv = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged
from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

flg = flag_invariant_brittleness(
    df,
    invariant_tags=list(inv),
    original_tag="original",
    metric_std_threshold=0.10,
)
st = brittleness_rate_from_flagged(flg)
print("Brittleness @ tau=0.10:", dict(st))

ts_csv_out = RES / "_colab_threshold_sensitivity_merged.csv"
ts_fig_out = RES / "_colab_threshold_sensitivity.pdf"
ts_proc = subprocess.run(
    [
        sys.executable,
        str(Path("/content/sbdd-robust/analysis/threshold_sensitivity.py")),
        "--metrics",
        str(metrics_csv),
        # threshold_sensitivity.py requires a second CSV; using the same DiffSBDD file
        # only satisfies the argparse contract — Pocket2Mol curve echoes DiffSBDD.
        "--pocket2mol-metrics",
        str(metrics_csv),
        "--thresholds",
        "0.03",
        "0.05",
        "0.08",
        "0.10",
        "0.15",
        "0.20",
        "--out-csv",
        str(ts_csv_out),
        "--out-figure",
        str(ts_fig_out),
    ],
    capture_output=True,
    text=True,
)
print(ts_proc.stdout[-24000:])
if ts_proc.stderr:
    print(ts_proc.stderr[-12000:])
if ts_proc.returncode != 0:
    print(
        "WARN threshold_sensitivity.py returned",
        ts_proc.returncode,
        "(figure backend issues are common; merged CSV may still exist).",
    )
if ts_csv_out.is_file():
    import pandas as pd2

    tdf_merge = pd2.read_csv(ts_csv_out)
    hit = tdf_merge[abs(tdf_merge["threshold"] - 0.10) < 1e-9]
    if hit.empty:
        print("WARN: τ=0.10 row missing from threshold_sensitivity CSV")
    else:
        row10 = hit.iloc[0]
        print(
            "threshold_sensitivity τ=0.10 brittleness_rate_diffsbdd:",
            row10.get("brittleness_rate_diffsbdd", float("nan")),
        )

sv_out = RES / "_colab_statistical_validation_out.csv"

subproc = subprocess.run(
    [
        sys.executable,
        str(Path("/content/sbdd-robust/analysis/statistical_validation.py")),
        "--metrics",
        str(metrics_csv),
        "--out",
        str(sv_out),
    ],
    capture_output=True,
    text=True,
)
print(subproc.stdout[-24000:])
if subproc.returncode != 0:
    print(subproc.stderr[-24000:])
    raise RuntimeError("statistical_validation.py failure")

picked: dict[str, dict] = {}
with sv_out.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        tnm = row.get("test") or ""
        ctr = row.get("contrast") or ""
        key = None
        if tnm == "paired_wilcoxon_validity_delta" and ctr == "original_vs_coordinate_jitter":
            key = "A. paired Wilcoxon (original vs jitter)"
        elif tnm == "onesample_t_mean_abs_validity_delta_jitter":
            key = "B. one-sample t (mean |Δvalid| jitter)"
        elif tnm == "wilcoxon_validity_sigma_across_four_stresses":
            key = "C. Wilcoxon on pocket dispersion sigma(validity)"
        if key:
            picked[key] = row

for k in ("A. paired Wilcoxon (original vs jitter)", "B. one-sample t (mean |Δvalid| jitter)", "C. Wilcoxon on pocket dispersion sigma(validity)"):
    print()
    print("===", k, "===")
    print(picked.get(k, {"missing": True}))


metas = sorted(RES.glob("run_meta__*.json"), key=lambda mp: mp.stat().st_mtime, reverse=True)
if metas:
    blob = json.loads(metas[0].read_text(encoding="utf-8"))
    print()
    print("run_meta brittleness_rate:", blob.get("brittleness_rate"))
else:
    print("No run_meta JSON found.")

print()
print("SUMMARY CELL FINISHED.")
""")


def build() -> dict:
    c: list[dict] = []

    c.append(
        md("""
### Cell 1 — GPU probe (**fail fast**)

Uses **`nvidia-smi`** (works **before** `pip install torch`) plus an optional Torch probe if imports exist. Throws if no NVIDIA driver binaries are reachable — switch to GPU runtime early.
""")
    )
    c.append(
        py("""
from __future__ import annotations

import shutil
import subprocess

if shutil.which("nvidia-smi") is None:
    raise RuntimeError(
        "nvidia-smi missing — Runtime → Change runtime type → GPU → Restart runtime."
    )

q = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    timeout=60,
)
if q.returncode != 0:
    raise RuntimeError(q.stderr[-4000:] or "nvidia-smi failed")
line = q.stdout.strip().splitlines()[0]
parts = [p.strip() for p in line.split(",")]
print("GPU:", parts[0] if parts else line)
try:
    print("VRAM MiB total / free:", parts[1], "/", parts[2])
except IndexError:
    pass

snap = subprocess.check_output(["nvidia-smi"], text=True, timeout=60)
print(snap[:3000])

# Post-install Torch sanity (cheap if wheel already cached)
try:
    import torch
except ImportError:
    print("(torch import skipped until Cell 3 finishes installing)")
else:
    if torch.cuda.is_available():
        print("torch.cuda OK:", torch.cuda.get_device_name(0), torch.version.cuda)
    else:
        print("WARN torch installed but CUDA not visible to PyTorch.")
""")
    )

    c.append(
        md("""
### Cell 2 — Mount Google Drive

Mounts Colab **`/content/drive`**, **`mkdir -p`** `MyDrive/sbdd-robust-results/real100`, binds **`DRIVE_OUT`**.""")
    )
    c.append(
        py("""
try:
    from google.colab import drive
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Not on Google Colab (missing google.colab). Upload this notebook to Colab "
        "or edit this cell."
    ) from exc

try:
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    raise RuntimeError(f"Drive mount failed: {exc}") from exc

from pathlib import Path

DRIVE_OUT = Path("/content/drive/MyDrive/sbdd-robust-results/real100")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print("DRIVE_OUT:", DRIVE_OUT.resolve())
""")
    )

    c.append(
        md("""
### Cell 3 — Pip installs (**CUDA 11.8** stack)

1. **`torch torchvision`** from pytorch cu118 wheels.
2. **`pytorch-lightning==1.8.4`** + **`torch-scatter`** from **`data.pyg.org`** (`torch-2.1.0+cu118` wheels).
3. Core chemistry/analysis wheels (**`rdkit`**, pinned **`openbabel-wheel`**, tables/plot libs, **`wandb>=0.16.6`**).  
   Older **`wandb==0.13.1`** (DiffSBDD conda parity) pulls unmaintained deps (**`pathtools`**, …) that often hit **`egg_info` / metadata-generation-failed** on Colab setuptools.

**Not installed here:** PyPI **`vina`** / **`meeko`** often fail on Colab images and **`compute_docking`** is **`false`** in `diffsbdd_real100.yaml` anyway. Enable docking later via `pip install meeko gemmi` + system **`vina`** (or `conda`) if you flip YAML.

Exports **`WANDB_MODE=offline`**. Pip failures embed pip stderr / stdout tails in **`RuntimeError`**.

""")
    )
    c.append(
        py("""
from __future__ import annotations

import os
import subprocess
import sys


def pip(parts: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *parts]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip()[-6000:])
    if proc.returncode != 0:
        nl = chr(10)
        tail = ""
        if proc.stderr and proc.stdout:
            tail = proc.stderr.strip() + nl + "--- pip stdout ---" + nl + proc.stdout.strip()
        elif proc.stderr:
            tail = proc.stderr.strip()
        elif proc.stdout:
            tail = proc.stdout.strip()
        tail = tail[-20000:] if tail else ""
        raise RuntimeError(
            "pip failed (exit "
            + str(proc.returncode)
            + "). Stderr/stdout tail: "
            + (tail or "(empty)")
        )


pip(
    [
        "install",
        "-q",
        "torch",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ]
)

pip([
    "install",
    "-q",
    "pytorch-lightning==1.8.4",
    "torch-scatter",
    "-f",
    "https://data.pyg.org/whl/torch-2.1.0+cu118.html",
])

# Modern setuptools wheels; avoids dependency chain trying legacy setup.py helpers.
pip(["install", "-q", "--upgrade", "pip", "setuptools", "wheel"])

pip(
    [
        "install",
        "-q",
        "rdkit",
        "biopython",
        "openbabel-wheel==3.1.1.23",
        "pyyaml",
        "tqdm",
        "wandb>=0.16.6,<1",
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "omegaconf",
        "networkx",
        "einops",
    ]
)

os.environ["WANDB_MODE"] = "offline"
os.environ.setdefault("WANDB_SILENT", "true")
print("WANDB_MODE =", os.environ["WANDB_MODE"])

import torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU vanished after reinstall — reinstall order / CUDA index mismatch.")

print("torch OK", torch.__version__)
""")
    )

    c.append(
        md(
            "\n".join(
                (
                    "### Cell 4 — Clone repos + **`pip install -e`**",
                    "",
                    "* DiffSBDD → **`/content/DiffSBDD`**  ",
                    "* **PocketBench** (`sbdd-robust`) → **`/content/sbdd-robust`**",
                    "",
                    "#### Private PocketBench on Colab",
                    "",
                    "HTTPS git cannot prompt interactively (no TTY on Colab). Do **one** of:",
                    "",
                    "1. **Colab Secrets / env:** we try **`GITHUB_TOKEN`**, **`GH_TOKEN`**, **`GITHUB_PAT`**, **`GIT_TOKEN`** "
                    "(env first, then Colab **`userdata`** with the same keys). Prefer **`GITHUB_TOKEN`**. "
                    "When Colab prompts, grant this notebook permission to read the secret. PAT: **Contents: Read** "
                    "(fine-grained) or **repo** (classic).",
                    "",
                    "2. **Full HTTPS URL:** set **`COLAB_SBDD_ROBUST_URL`** to "
                    "**`https://x-access-token:PAT@github.com/OWNER/repo.git`**.",
                    "",
                    "3. **Different fork/org:** **`PRIVATE_POCKETBENCH_SLUG`** = **`OWNER/repo`** "
                    "(default **`Marooncoloredchair/PocketBench`**).",
                    "",
                    "4. **Stale partial clone:** delete **`/content/sbdd-robust`** via the Files sidebar, then rerun Cell 4.",
                )
            )
        )
    )
    c.append(
        py("""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def redact_https_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    if "@" not in parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port:
        host += ":" + str(parts.port)
    netloc_safe = "***@" + host
    return urlunsplit((parts.scheme, netloc_safe, parts.path, parts.query, parts.fragment))


def colab_github_credential() -> tuple[str, str]:
    # Returns (token, source_label); never log the raw token.
    keys = ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT", "GIT_TOKEN")
    for k in keys:
        t = (os.environ.get(k) or "").strip()
        if t:
            return t, "environment " + k
    try:
        from google.colab import userdata as _userdata  # type: ignore

        for k in keys:
            t = (_userdata.get(k) or "").strip()
            if t:
                return t, "Colab secret " + k
    except Exception:
        pass
    return "", ""


def pocketbench_clone_url() -> str:
    explicit = (os.environ.get("COLAB_SBDD_ROBUST_URL") or "").strip()
    if explicit:
        return explicit
    tok, _src = colab_github_credential()
    slug = (
        os.environ.get("PRIVATE_POCKETBENCH_SLUG", "Marooncoloredchair/PocketBench")
        .strip()
        .strip("/")
    )
    if slug.endswith(".git"):
        slug = slug[:-4]
    if not tok:
        return "https://github.com/" + slug + ".git"
    return "https://x-access-token:" + tok + "@github.com/" + slug + ".git"


def clone(url: str, dest: Path) -> None:
    if (dest / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only"],
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        print("reuse", dest)
        return
    if dest.exists():
        raise RuntimeError(str(dest) + " occupied — delete via Files sidebar")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    print("$ git clone", redact_https_url(url), "->", str(dest))
    r = subprocess.run(
        [
            "git",
            "-c",
            "credential.helper=",
            "clone",
            "--depth",
            "1",
            url,
            str(dest),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        nl = chr(10)
        ta = ((r.stderr or "").rstrip() + nl + (r.stdout or "").rstrip()).strip()
        tail = ta[-9000:] if ta else "(no git output)"
        hint = ""
        purl = urlsplit(url)
        if (
            (purl.scheme or "").lower() == "https"
            and "github.com" in (purl.netloc or "").lower()
            and "@" not in (purl.netloc or "")
            and not colab_github_credential()[0]
        ):
            hint = (
                nl
                + "(no token) Set Secret GITHUB_TOKEN (or GH_TOKEN / GITHUB_PAT) or COLAB_SBDD_ROBUST_URL."
            )
        raise RuntimeError(
            "git clone failed:"
            + nl
            + redact_https_url(url)
            + " -> "
            + str(dest)
            + nl
            + "Private repo: Secrets -> add GITHUB_TOKEN (Contents Read), "
            + "or set COLAB_SBDD_ROBUST_URL; wipe partial clone and retry."
            + hint
            + nl
            + "--- git output ---"
            + nl
            + tail
        )


_pb_url = pocketbench_clone_url()
_pb_parts = urlsplit(_pb_url)
_pb_auth = "@" in (_pb_parts.netloc or "")
_tok_vis, _tok_src = colab_github_credential()
_explicit_pb = bool((os.environ.get("COLAB_SBDD_ROBUST_URL") or "").strip())

if _pb_auth:
    if _explicit_pb:
        print(
            "PocketBench: COLAB_SBDD_ROBUST_URL set (credentials masked in clone log):",
            redact_https_url(_pb_url),
        )
    elif _tok_vis:
        print("PocketBench: PAT from", _tok_src + " (value not printed)")
    else:
        print("PocketBench:", redact_https_url(_pb_url))
elif (_pb_parts.scheme or "").lower() == "https" and "github.com" in (
    (_pb_parts.netloc or "").lower()
):
    print(
        "PocketBench: anonymous github.com URL — for a private repo add a Colab Secret "
        "GITHUB_TOKEN (or GH_TOKEN / GITHUB_PAT / GIT_TOKEN) or set COLAB_SBDD_ROBUST_URL."
    )

clone(os.environ.get("COLAB_DIFFSBDD_URL", "https://github.com/arneschneuing/DiffSBDD.git"),
      Path("/content/DiffSBDD"))

clone(_pb_url, Path("/content/sbdd-robust"))

SBDD_ROOT = Path("/content/sbdd-robust").resolve()
_pkg_init = SBDD_ROOT / "sbdd_robust" / "__init__.py"
if not _pkg_init.is_file():
    raise RuntimeError(
        "PocketBench checkout is missing "
        + str(_pkg_init)
        + " — wrong PRIVATE_POCKETBENCH_SLUG, shallow clone glitch, "
        + "or a fork/publish layout without the benchmarking package tree."
    )

# Editable install registers metadata/console entry points. Deps satisfied in Cell 3.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(SBDD_ROOT),
        "--no-deps",
    ]
)

# Notebook cwd on Colab is usually /content, not SBDD_ROOT. Editable installs add a .pth
# entry when pip succeeds, but kernels occasionally miss it until reload; prepend repo root.
_root_s = str(SBDD_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

import importlib

importlib.invalidate_caches()
import sbdd_robust  # noqa

print("sbdd_robust ready (repo", str(SBDD_ROOT) + ")")
""")
    )

    c.append(
        md("""
### Cell 5 — Checkpoint **Zenodo 8183747**

Downloads **`crossdocked_fullatom_cond.ckpt`** into **`/content/DiffSBDD/checkpoints`** (urllib fallback wget). Validates size **> 1 MB**""")
    )
    c.append(
        py("""
from pathlib import Path
import subprocess
import urllib.request

URL = "https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1"
folder = Path("/content/DiffSBDD/checkpoints")
folder.mkdir(parents=True, exist_ok=True)
dst = folder / "crossdocked_fullatom_cond.ckpt"

if not (dst.is_file() and dst.stat().st_size > 1_000_000):
    try:
        urllib.request.urlretrieve(URL, dst)
    except Exception:
        subprocess.check_call(["wget", "-q", "-O", str(dst), URL])

if dst.stat().st_size < 1_000_000:
    raise RuntimeError("checkpoint too small / corrupt")
print("checkpoint bytes", dst.stat().st_size)
""")
    )

    c.append(
        md(
            "\n".join(
                (
                    "### Cell 6 - **Real-100 PDBs + `diffsbdd_real100.yaml`**",
                    "",
                    "1. Preferred: use the **committed `configs/diffsbdd_real100.yaml`** (the frozen panel) and "
                    "**fetch only missing PDBs** from **`files.rcsb.org`**. The generator is *not* re‑run here, "
                    "so the panel always matches the committed PDBs.",
                    "",
                    "2. If the YAML is absent (partial clone) but **`scripts/generate_real100_config.py`** exists, "
                    "Cell 6 runs the generator (heavy RCSB Search API filtering).",
                    "",
                    "3. If both are missing, Cell 6 restores **`configs/diffsbdd_real100.yaml`** "
                    "from a zlib/base64 blob embedded when **`colab/make_benchmark_nb.py`** was executed, "
                    "then PDB downloads plus count check **`>= 100` PDBs**.",
                )
            )
        )
    )
    _payload = bundled_diffsbdd_real100_zlib_b64()
    c.append(py(_CELL6_CODE_TEMPLATE.replace("@@DIFFSBDD_REAL100_ZLIB_B64@@", _payload)))
    c.append(
        md(
            "### Cell 7 — DiffSBDD upstream patches (**Colab**)\n\n"
            "Inline edits under **`/content/DiffSBDD`**: `torch.load`, `--device`, "
            "Biopython `three_to_one` shim, **`\\pm`** escape in `metrics.py`, "
            "insertion‑code pocket residues, **`SDWriter`** close."
        )
    )
    c.append(py(CELL7.strip("\n")))
    c.append(
        md(
            "### Cell 8 — `diffsbdd_real100_colab.yaml`\n\n"
            "Writes **`/content/sbdd-robust/configs/diffsbdd_real100_colab.yaml`** with Colab literals, "
            "**`skip_failed_pockets: true`**, **`extra_args: [--device, cuda]`**, and a stable "
            "**`run_id: real100`** + **`resume: true`**. When **Drive is mounted (Cell 2)**, "
            "**`paths.results`** / **`paths.generations`** point at "
            "**`MyDrive/sbdd-robust-results/real100/work`** so every finished pocket is checkpointed "
            "to Drive — a disconnect mid‑run is recoverable by just re‑running Cell 9."
        )
    )
    c.append(py(CELL8.strip("\n")))
    c.append(
        md(
            "### Cell 9 — Benchmark (**streamed, resumable**)\n\n"
            "Exports **`DIFFSBDD_*`** + **`PYTHONUNBUFFERED`**, runs with **`--resume`**, "
            "**`Popen` line streaming** (**merged stderr**). 100 pockets × 5 perturbations is "
            "**many hours** — if Colab disconnects, just **re‑run this cell**; completed "
            "(pocket, perturbation) conditions in the Drive checkpoint CSV are **skipped**."
        )
    )
    c.append(py(CELL9.strip("\n")))
    c.append(
        md(
            "### Cell 10 — Snapshot to Drive\n\n"
            "Reads the results dir from the Cell 8 sidecar. If results already live on Drive "
            "(checkpointed during the run), it just verifies; otherwise it mirrors into **`DRIVE_OUT`**."
        )
    )
    c.append(py(CELL10.strip("\n")))
    c.append(
        md(
            "### Cell 11 — Statistical digest\n\n"
            "**Brittleness @ τ = 0.10** plus **coverage** (original **`n_valid>0`** row fraction).\n\n"
            "Runs **`analysis/threshold_sensitivity.py`** subprocess (duplicate metrics path satisfies the "
            "required Pocket2Mol slot so only the DiffSBDD curve matters) and parses **τ = 0.10** "
            "**`brittleness_rate_diffsbdd`** from **`_colab_threshold_sensitivity_merged.csv`**.\n\n"
            "Runs **`analysis/statistical_validation.py`**, echoes the CSV rows for the paired "
            "Wilcoxon (jitter), one-sample *t*, and dispersion Wilcoxon, then prints **`run_meta.brittleness_rate`**."
        )
    )
    c.append(py(CELL11.strip("\n")))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "collapsed_sections": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": c,
    }


def squash_code_cell_sources(nb: dict) -> None:
    """Join split `source` fragments so serializers are less likely to break mid-line escapes."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if isinstance(src, list):
            cell["source"] = ["".join(src)]


def assert_notebook_code_parses(nb: dict) -> None:
    import ast

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        blob = "".join(src) if isinstance(src, list) else src
        try:
            ast.parse(blob, filename=f"cell-{i}")
        except SyntaxError as exc:
            raise SyntaxError(f"Notebook code cell index {i} failed to parse") from exc


def main() -> None:
    out = Path(__file__).resolve().parent / "sbdd_robust_benchmark.ipynb"
    nb = build()
    squash_code_cell_sources(nb)
    assert_notebook_code_parses(nb)
    out.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
