#!/usr/bin/env python3
r"""Build ``pocket2mol_real100_benchmark.ipynb`` for Pocket2Mol real100 (Colab).

Run locally:

    python colab/make_pocket2mol_nb.py

Design notes
------------
Pocket2Mol's published stack is **Python 3.8 / PyTorch 1.10.1 / CUDA 11.3 /
PyG 2.0.4** (see ``env_cuda113.yml`` in ``pengxingang/Pocket2Mol``). That cannot
coexist with Colab's modern base Python, so the model runs inside an **isolated
micromamba env** while ``sbdd_robust`` orchestrates from base Python. The adapter
indirection (``model.python_exe`` / ``model.script_path``) makes this clean: the
driver only ever reads the per-pocket ``SMILES.txt`` sidecar, never unpickles a
Pocket2Mol ``.pt`` (so base Python needs no torch).

The notebook is **resumable**: results are checkpointed to Google Drive, and
re-running the run cell skips finished (pocket, perturbation) conditions.
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


def bundled_pocket2mol_real100_zlib_b64() -> str:
    """zlib+base64 of configs/pocket2mol_real100.yaml so Cell 6 can restore the frozen panel."""
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "pocket2mol_real100.yaml"
    blob = zlib.compress(yaml_path.read_bytes(), level=9)
    return base64.standard_b64encode(blob).decode("ascii")


# --------------------------------------------------------------------------------------
# Stable literals shared across cells.
# --------------------------------------------------------------------------------------
P2M_PREFIX = "/content/micromamba/envs/p2m"
P2M_PYTHON = P2M_PREFIX + "/bin/python"
P2M_REPO = "/content/Pocket2Mol"
MAMBA_ROOT = "/content/micromamba"
MAMBA_BIN = "/content/bin/micromamba"
RUN_ID = "pocket2mol_real100"
DRIVE_SUBDIR = "pocket2mol_real100"

# Pocket2Mol runtime packages (MKL/blas removal can drop these from the p2m env).
P2M_RESTORE_SPECS = [
    "numpy=1.22.3",
    "pyyaml=5.4.1",
    "python-dateutil",
    "pytz",
    "joblib",
    "threadpoolctl",
    "biopython=1.79",
    "easydict=1.9",
    "rdkit=2022.03.2",
    "python-lmdb=1.2.1",
    "tqdm",
]
P2M_RUNTIME_PROBE = (
    "import numpy, yaml; "
    "from Bio import BiopythonWarning; "
    "from easydict import EasyDict; "
    "from rdkit import Chem; "
    "import dateutil, pytz, joblib, threadpoolctl; "
    "print('p2m runtime OK')"
)


# --------------------------------------------------------------------------------------
# Cell 6 (PDBs + config restore) — string template, fills in the embedded YAML blob.
# --------------------------------------------------------------------------------------
_CELL6_CODE_TEMPLATE = '''
from __future__ import annotations

import base64
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

ROOT = Path("/content/sbdd-robust").resolve()
CFG = ROOT / "configs" / "pocket2mol_real100.yaml"
RAW = ROOT / "data/raw/real100"
RAW.mkdir(parents=True, exist_ok=True)

# Restore the frozen 100-pocket config from an embedded blob if a partial clone
# dropped it (the committed PDBs always match this panel).
if not CFG.is_file():
    blob = zlib.decompress(base64.standard_b64decode("@@POCKET2MOL_REAL100_ZLIB_B64@@"))
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_bytes(blob)
    print("restored", CFG, "from embedded blob")

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML missing -- rerun Cell 3") from exc

with CFG.open(encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
pockets = cfg.get("pockets") if isinstance(cfg, dict) else None
if not isinstance(pockets, list) or not pockets:
    raise RuntimeError(str(CFG) + " has no pockets:[]")

fetched = 0
for p in pockets:
    if not isinstance(p, dict):
        continue
    rel = Path(str(p.get("pdb") or "").strip()).name.lower()
    if not rel.endswith(".pdb"):
        raise RuntimeError("invalid pdb entry: " + str(p))
    pid = rel[:-4].strip().upper()
    if len(pid) != 4 or not pid.isalnum():
        raise RuntimeError("unexpected PDB ID: " + repr(pid))
    dest = RAW / rel
    if dest.is_file() and dest.stat().st_size > 500:
        continue
    url = "https://files.rcsb.org/download/" + urllib.parse.quote(pid) + ".pdb"
    urllib.request.urlretrieve(url, dest)
    fetched += 1

have = sorted(RAW.glob("*.pdb"))
print("PDBs present:", len(have), "(fetched", fetched, "this run)")
if len(have) < len(pockets):
    raise RuntimeError(
        "Expected " + str(len(pockets)) + " PDBs, found " + str(len(have))
        + " -- re-run this cell (RCSB rate limit / transient network)."
    )
'''


def build() -> dict:
    c: list[dict] = []

    # ---- Title ----------------------------------------------------------------
    c.append(
        md(
            """
# Pocket2Mol real100 robustness benchmark (Colab)

Runs the **PocketBench** reliability suite (featurization-noise + pocket-boundary
perturbations -> normalized brittleness, paired Wilcoxon crop test, **PBSI**) on the
frozen **100-pocket** panel using **Pocket2Mol** (`pengxingang/Pocket2Mol`).

**Architecture.** Pocket2Mol needs an old stack (Python 3.8 / PyTorch 1.10.1 /
CUDA 11.3 / PyG 2.0.4) that cannot live in Colab's base Python, so the model runs
inside an isolated **micromamba** env (`p2m`) while `sbdd_robust` orchestrates from
base Python. Results checkpoint to **Google Drive**, so the run is fully resumable:
if Colab disconnects, just re-run the run cell.

**Prereqs.**
- GPU runtime (Runtime -> Change runtime type -> GPU).
- Drive holds the checkpoint at `MyDrive/pocket2mol_ckpt/pretrained_Pocket2Mol.pt`.
- A Colab Secret `GITHUB_TOKEN` (Contents: Read) for the private PocketBench repo.
"""
        )
    )

    # ---- Cell 1: GPU probe ----------------------------------------------------
    c.append(
        md(
            """
### Cell 1 — GPU probe (**fail fast**)

Uses `nvidia-smi` (works before any pip/conda install). Throws if no NVIDIA driver
is reachable — switch to a GPU runtime first.
"""
        )
    )
    c.append(
        py(
            """
from __future__ import annotations

import shutil
import subprocess

if shutil.which("nvidia-smi") is None:
    raise RuntimeError(
        "nvidia-smi missing -- Runtime -> Change runtime type -> GPU -> Restart runtime."
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
print(subprocess.check_output(["nvidia-smi"], text=True, timeout=60)[:3000])
"""
        )
    )

    # ---- Cell 2: Drive --------------------------------------------------------
    c.append(
        md(
            f"""
### Cell 2 — Mount Google Drive

Mounts `/content/drive`, creates `MyDrive/sbdd-robust-results/{DRIVE_SUBDIR}`,
binds `DRIVE_OUT`. Results checkpoint here so a disconnect mid-run is recoverable.
"""
        )
    )
    c.append(
        py(
            f"""
try:
    from google.colab import drive
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Not on Google Colab (missing google.colab). Upload this notebook to Colab."
    ) from exc

try:
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    raise RuntimeError(f"Drive mount failed: {{exc}}") from exc

from pathlib import Path

DRIVE_OUT = Path("/content/drive/MyDrive/sbdd-robust-results/{DRIVE_SUBDIR}")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print("DRIVE_OUT:", DRIVE_OUT.resolve())
"""
        )
    )

    # ---- Cell 3: base-python orchestrator deps --------------------------------
    c.append(
        md(
            """
### Cell 3 — Orchestrator deps (**base Python, no torch**)

`sbdd_robust` only orchestrates Pocket2Mol here and reads each pocket's `SMILES.txt`
sidecar, so base Python needs **no GPU / no torch** — just chemistry + analysis
wheels. The heavy Pocket2Mol stack is installed separately in Cell 5.
"""
        )
    )
    c.append(
        py(
            """
from __future__ import annotations

import subprocess
import sys


def pip(parts: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", *parts]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip()[-4000:])
    if proc.returncode != 0:
        nl = chr(10)
        tail = ((proc.stderr or "") + nl + (proc.stdout or "")).strip()[-16000:]
        raise RuntimeError("pip failed (exit " + str(proc.returncode) + "): " + (tail or "(empty)"))


pip(["install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
pip(
    [
        "install",
        "-q",
        "rdkit",
        "biopython",
        "pyyaml",
        "tqdm",
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "omegaconf",
        "networkx",
    ]
)
print("orchestrator deps OK")
"""
        )
    )

    # ---- Cell 4: clone PocketBench + Pocket2Mol -------------------------------
    c.append(
        md(
            "\n".join(
                (
                    "### Cell 4 — Clone repos + `pip install -e` PocketBench",
                    "",
                    "* **PocketBench** (`sbdd-robust`, private) → `/content/sbdd-robust`",
                    "* **Pocket2Mol** (`pengxingang/Pocket2Mol`) → `/content/Pocket2Mol`",
                    "",
                    "Private PocketBench needs a token (no TTY on Colab). We try env / Colab "
                    "Secrets `GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_PAT`, `GIT_TOKEN`; or set "
                    "`COLAB_SBDD_ROBUST_URL` to a full `https://x-access-token:PAT@github.com/OWNER/repo.git`. "
                    "Override the slug via `PRIVATE_POCKETBENCH_SLUG` (default "
                    "`Marooncoloredchair/PocketBench`).",
                )
            )
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def redact_https_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port:
        host += ":" + str(parts.port)
    return urlunsplit((parts.scheme, "***@" + host, parts.path, parts.query, parts.fragment))


def colab_github_credential() -> tuple[str, str]:
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
    tok, _ = colab_github_credential()
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
            env={{**os.environ, "GIT_TERMINAL_PROMPT": "0"}},
        )
        print("reuse", dest)
        return
    if dest.exists():
        raise RuntimeError(str(dest) + " occupied -- delete via Files sidebar")
    env = {{**os.environ, "GIT_TERMINAL_PROMPT": "0"}}
    print("$ git clone", redact_https_url(url), "->", str(dest))
    r = subprocess.run(
        ["git", "-c", "credential.helper=", "clone", "--depth", "1", url, str(dest)],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        nl = chr(10)
        tail = ((r.stderr or "").rstrip() + nl + (r.stdout or "").rstrip()).strip()[-9000:]
        raise RuntimeError(
            "git clone failed: " + redact_https_url(url) + " -> " + str(dest) + nl
            + "Private repo: add Colab Secret GITHUB_TOKEN (Contents: Read) or set "
            + "COLAB_SBDD_ROBUST_URL; wipe any partial clone and retry." + nl
            + "--- git output ---" + nl + (tail or "(no git output)")
        )


_pb_url = pocketbench_clone_url()
_tok, _src = colab_github_credential()
if "@" in (urlsplit(_pb_url).netloc or "") and _tok:
    print("PocketBench: PAT from", _src, "(value not printed)")

clone("https://github.com/pengxingang/Pocket2Mol.git", Path("{P2M_REPO}"))
clone(_pb_url, Path("/content/sbdd-robust"))

SBDD_ROOT = Path("/content/sbdd-robust").resolve()
if not (SBDD_ROOT / "sbdd_robust" / "__init__.py").is_file():
    raise RuntimeError(
        "PocketBench checkout missing sbdd_robust/__init__.py -- wrong slug or partial clone."
    )

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(SBDD_ROOT), "--no-deps"]
)
if str(SBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(SBDD_ROOT))
importlib.invalidate_caches()
import sbdd_robust  # noqa

if not (Path("{P2M_REPO}") / "sample_for_pdb.py").is_file():
    raise RuntimeError(
        "Pocket2Mol checkout missing sample_for_pdb.py at {P2M_REPO}."
    )
print("sbdd_robust ready; Pocket2Mol repo at {P2M_REPO}")
"""
        )
    )

    # ---- Cell 5: micromamba env (single-solve, no post-create surgery) --------
    c.append(
        md(
            """
### Cell 5 — Isolated **Pocket2Mol** env via micromamba (**~5-10 min**)

Pocket2Mol's published stack (`env_cuda113.yml`) is **Python 3.8 / PyTorch 1.10.1 /
CUDA 11.3 / PyG 2.0.4** — incompatible with Colab's base Python. We install
**micromamba** (no kernel restart) and create env `p2m` in **one solve** from the
**pytorch / pyg / conda-forge** channels. `cudatoolkit=11.3` bundles its own CUDA
runtime, so it only needs the NVIDIA driver (already present on a GPU runtime).

**MKL is pinned to `2021.4.0` at create time.** PyTorch 1.10 links Intel MKL's
`iJIT_NotifyEvent`, which MKL ≥2022 dropped (`ImportError: ... undefined symbol:
iJIT_NotifyEvent`). Pinning up front lets the solver build **one consistent
environment** — we never remove/swap packages afterward (that orphaned numpy /
biopython / torch in earlier attempts).

This cell is **idempotent and self-healing**: it skips create if the env imports
cleanly, and **wipes + rebuilds** if anything is broken. Fallback path creates from
the upstream `env_cuda113.yml`.
"""
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MAMBA_ROOT = Path("{MAMBA_ROOT}")
MAMBA_BIN = Path("{MAMBA_BIN}")
P2M_PREFIX = Path("{P2M_PREFIX}")
P2M_PYTHON = Path("{P2M_PYTHON}")

# Full import probe: torch + pyg + everything sample_for_pdb.py / the bridge need.
_PROBE = (
    "import torch, torch_geometric, numpy, yaml; "
    "from Bio import BiopythonWarning; "
    "from easydict import EasyDict; "
    "from rdkit import Chem; "
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
    "'avail', torch.cuda.is_available()); "
    "print('pyg', torch_geometric.__version__)"
)

# One-solve spec. mkl=2021.4.0 pinned so torch 1.10 imports; all runtime deps included
# so nothing is added/removed after create (avoids orphaning numpy/biopython/torch).
_CREATE_CHANNELS = ["-c", "pytorch", "-c", "pyg", "-c", "conda-forge"]
_CREATE_SPECS = [
    "python=3.8",
    "pytorch=1.10.1",
    "cudatoolkit=11.3",
    "pytorch-mutex=1.0=cuda",
    "mkl=2021.4.0",
    "pyg=2.0.4",
    "pytorch-scatter=2.0.9",
    "pytorch-sparse=0.6.13",
    "pytorch-cluster=1.6.0",
    "pytorch-spline-conv=1.2.1",
    "rdkit=2022.03.2",
    "biopython=1.79",
    "easydict=1.9",
    "python-lmdb=1.2.1",
    "pyyaml=5.4.1",
    "numpy=1.22.3",
    "pandas=1.4.2",
    "scikit-learn=1.1.0",
    "scipy=1.8.0",
    "networkx=2.8",
    "python-dateutil",
    "pytz",
    "joblib",
    "threadpoolctl",
    "tqdm",
]


def _run(cmd: list[str], timeout: int = 3600) -> None:
    print("$", " ".join(cmd))
    env = {{**os.environ, "MAMBA_ROOT_PREFIX": str(MAMBA_ROOT)}}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if proc.wait(timeout=timeout) != 0:
        raise RuntimeError("command failed: " + " ".join(cmd))


def _probe() -> subprocess.CompletedProcess:
    return subprocess.run([str(P2M_PYTHON), "-c", _PROBE], capture_output=True, text=True)


def _create_explicit() -> None:
    _run([str(MAMBA_BIN), "create", "-y", "-p", str(P2M_PREFIX), *_CREATE_CHANNELS, *_CREATE_SPECS])


def _create_from_yml() -> None:
    yml = Path("/content/Pocket2Mol/env_cuda113.yml")
    if not yml.is_file():
        raise FileNotFoundError("Missing " + str(yml) + " -- run Cell 4 (clone Pocket2Mol) first.")
    print("Fallback: creating env from upstream env_cuda113.yml ...")
    _run([str(MAMBA_BIN), "create", "-y", "-p", str(P2M_PREFIX), "-f", str(yml)])
    # env_cuda113.yml omits a couple deps the bridge path touches.
    _run([str(MAMBA_BIN), "install", "-y", "-p", str(P2M_PREFIX), "-c", "conda-forge", "easydict=1.9"])


def _build_env() -> None:
    shutil.rmtree(P2M_PREFIX, ignore_errors=True)
    try:
        _create_explicit()
    except RuntimeError as exc:
        print("Explicit single-solve create failed:", exc)
        shutil.rmtree(P2M_PREFIX, ignore_errors=True)
        _create_from_yml()


# 1) micromamba binary
if not MAMBA_BIN.is_file():
    MAMBA_BIN.parent.mkdir(parents=True, exist_ok=True)
    print("Installing micromamba ...")
    dl = subprocess.run(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xvj -C /content bin/micromamba",
        shell=True,
        capture_output=True,
        text=True,
    )
    if dl.returncode != 0 or not MAMBA_BIN.is_file():
        raise RuntimeError("micromamba download failed:\\n" + (dl.stderr or dl.stdout)[-4000:])
print("micromamba:", MAMBA_BIN)

# 2) build env if missing/broken. No post-create install/remove: rebuild from scratch.
_need_build = True
if P2M_PYTHON.is_file():
    chk = _probe()
    if chk.returncode == 0:
        print("env exists and imports cleanly, skipping create:", P2M_PYTHON)
        _need_build = False
    else:
        print("Existing env is broken -- rebuilding from scratch. Probe error tail:")
        print((chk.stderr or chk.stdout)[-1500:])

if _need_build:
    _build_env()

# 3) verify (forces a CUDA context + imports every package the run needs)
probe = _probe()
print(probe.stdout)
if probe.returncode != 0:
    raise RuntimeError("Pocket2Mol env verification failed:\\n" + probe.stderr[-6000:])
if "avail True" not in probe.stdout:
    print("WARN: torch.cuda.is_available() is False in the p2m env -- the run will be CPU-only/slow.")
print("Pocket2Mol env ready:", P2M_PYTHON)
"""
        )
    )

    # ---- Cell 6: PDBs + frozen config ----------------------------------------
    c.append(
        md(
            """
### Cell 6 — Real-100 PDBs + frozen `pocket2mol_real100.yaml`

Uses the committed `configs/pocket2mol_real100.yaml` (restoring it from an embedded
blob if a partial clone dropped it) and fetches any missing PDBs from `files.rcsb.org`.
"""
        )
    )
    payload = bundled_pocket2mol_real100_zlib_b64()
    c.append(py(_CELL6_CODE_TEMPLATE.replace("@@POCKET2MOL_REAL100_ZLIB_B64@@", payload)))

    # ---- Cell 7: locate checkpoint + write colab config -----------------------
    c.append(
        md(
            f"""
### Cell 7 — Locate checkpoint + write `pocket2mol_real100_colab.yaml`

Finds `pretrained_Pocket2Mol.pt` (default `MyDrive/pocket2mol_ckpt/`), then writes a
Colab config with **Drive-backed** `paths.results/generations`, `run_id={RUN_ID}`,
`resume: true`, `normalized_only: true`, and concrete `model` paths
(`python_exe` → the `p2m` env, `script_path` → the bridge). A sidecar `.pb_paths.json`
keeps later cells in sync regardless of run order.
"""
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path("/content/sbdd-robust")
src = ROOT / "configs" / "pocket2mol_real100.yaml"
if not src.is_file():
    raise RuntimeError("Run Cell 6 first (missing " + str(src) + ").")

# Locate the checkpoint (Drive first; user keeps it at MyDrive/pocket2mol_ckpt/).
_ckpt_candidates = [
    Path("/content/drive/MyDrive/pocket2mol_ckpt/pretrained_Pocket2Mol.pt"),
    Path("{P2M_REPO}/ckpt/pretrained_Pocket2Mol.pt"),
    Path("/content/drive/MyDrive/pretrained_Pocket2Mol.pt"),
]
CKPT = next((p for p in _ckpt_candidates if p.is_file()), None)
if CKPT is None:
    raise RuntimeError(
        "pretrained_Pocket2Mol.pt not found. Looked in:\\n  "
        + "\\n  ".join(str(p) for p in _ckpt_candidates)
        + "\\nUpload it to MyDrive/pocket2mol_ckpt/ or set the path here."
    )
print("checkpoint:", CKPT, "(", CKPT.stat().st_size, "bytes )")

P2M_PYTHON = "{P2M_PYTHON}"
if not Path(P2M_PYTHON).is_file():
    raise RuntimeError("Pocket2Mol env Python missing (" + P2M_PYTHON + ") -- run Cell 5.")
BRIDGE = (ROOT / "scripts" / "pocket2mol_sample_drug_bridge.py").resolve()
if not BRIDGE.is_file():
    raise RuntimeError("Bridge script missing: " + str(BRIDGE))

with src.open(encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
if DRIVE_MYDRIVE.is_dir():
    WORK = DRIVE_MYDRIVE / "sbdd-robust-results" / "{DRIVE_SUBDIR}" / "work"
    on_drive = True
else:
    WORK = Path("/content/sbdd_robust_work/data")
    on_drive = False
RESULTS_DIR = WORK / "results"
GENERATIONS_DIR = WORK / "generations"
for d in (RESULTS_DIR, GENERATIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

cfg["project_root"] = str(ROOT)
cfg.setdefault("paths", {{}})
cfg["paths"]["results"] = str(RESULTS_DIR)
cfg["paths"]["generations"] = str(GENERATIONS_DIR)
cfg["skip_failed_pockets"] = True
cfg["run_id"] = "{RUN_ID}"
cfg["resume"] = True
cfg["normalized_only"] = True

m = cfg.setdefault("model", {{}})
m["type"] = "pocket2mol"
m["repo_root"] = "{P2M_REPO}"
m["script_path"] = str(BRIDGE)
m["checkpoint"] = str(CKPT)
m["python_exe"] = P2M_PYTHON
m.setdefault("n_samples", 20)
m["sanitize"] = True
m["extra_args"] = ["--device", "cuda"]

outp = ROOT / "configs" / "pocket2mol_real100_colab.yaml"
with outp.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, width=140)

(ROOT / ".pb_paths.json").write_text(
    json.dumps(
        {{
            "results": str(RESULTS_DIR),
            "generations": str(GENERATIONS_DIR),
            "run_id": "{RUN_ID}",
            "on_drive": on_drive,
            "checkpoint": str(CKPT),
            "p2m_python": P2M_PYTHON,
        }}
    ),
    encoding="utf-8",
)
print("wrote", outp)
print("results dir:", RESULTS_DIR, "(Drive)" if on_drive else "(EPHEMERAL /content)")
if not on_drive:
    print("WARNING: Drive not mounted (run Cell 2) -- results will be LOST on disconnect.")
"""
        )
    )

    # ---- Cell 8: bridge smoke test (1 pocket) ---------------------------------
    c.append(
        md(
            """
### Cell 8 — Bridge smoke test (**1 pocket, 3 samples**)

Validates the whole Pocket2Mol chain (env → `sample_for_pdb.py` → bridge →
`samples_all.pt` + `SMILES.txt`) on a single pocket **before** the multi-hour run, so
env problems surface in ~2 minutes instead of mid-benchmark.
"""
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/content/sbdd-robust")
info = json.loads((ROOT / ".pb_paths.json").read_text(encoding="utf-8"))
CKPT = info["checkpoint"]
P2M_PYTHON = info["p2m_python"]
BRIDGE = ROOT / "scripts" / "pocket2mol_sample_drug_bridge.py"

# Verify the p2m env has everything sample_for_pdb.py needs (don't mutate it here --
# Cell 5 owns env construction so we never re-introduce an MKL/torch mismatch).
_runtime_probe = {repr(P2M_RUNTIME_PROBE)}
_dep = subprocess.run([P2M_PYTHON, "-c", _runtime_probe], capture_output=True, text=True)
if _dep.returncode != 0:
    raise RuntimeError(
        "p2m env is missing runtime deps -- re-run Cell 5 (it builds the env in one solve).\\n"
        + (_dep.stderr or _dep.stdout)[-1500:]
    )

# First committed pocket PDB as the probe input.
import yaml

with (ROOT / "configs" / "pocket2mol_real100.yaml").open(encoding="utf-8") as fh:
    pockets = yaml.safe_load(fh)["pockets"]
pdb = (ROOT / str(pockets[0]["pdb"])).resolve()
if not pdb.is_file():
    raise RuntimeError("probe PDB missing: " + str(pdb) + " -- run Cell 6.")

with tempfile.TemporaryDirectory() as td:
    out_pt = Path(td) / "smoke_out.pt"
    cmd = [
        P2M_PYTHON, str(BRIDGE),
        "--pdb_path", str(pdb),
        "--num_samples", "3",
        "--result_path", str(out_pt),
        "--checkpoint", str(CKPT),
        "--device", "cuda",
        "--pocket2mol_root", "{P2M_REPO}",
    ]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout[-3000:])
    if proc.returncode != 0:
        tail = ((proc.stderr or "").rstrip() + "\\n" + (proc.stdout or "").rstrip()).strip()
        raise RuntimeError("bridge smoke test failed:\\n" + tail[-6000:])
    sidecar = out_pt.with_name(out_pt.stem + "_smiles.txt")
    n = len(sidecar.read_text(encoding="utf-8").split()) if sidecar.is_file() else 0
    print("smoke test OK:", out_pt.name, "exists;", n, "SMILES in sidecar")
"""
        )
    )

    # ---- Cell 9: run ----------------------------------------------------------
    c.append(
        md(
            """
### Cell 9 — Benchmark (**streamed, resumable**)

Runs `pocketbench run --resume --normalized-only`. 100 pockets × 5 perturbations ×
20 samples is **many hours** — if Colab disconnects, just **re-run this cell**;
finished (pocket, perturbation) conditions in the Drive checkpoint CSV are skipped.
"""
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/sbdd-robust")
cfgp = ROOT / "configs" / "pocket2mol_real100_colab.yaml"
if not cfgp.is_file():
    raise RuntimeError("Run Cell 7 first.")

info = json.loads((ROOT / ".pb_paths.json").read_text(encoding="utf-8"))
res_dir = Path(info["results"])
run_id = info.get("run_id", "{RUN_ID}")
prior = res_dir / ("metrics_per_condition__run" + run_id + ".csv")
print("results dir:", res_dir, "(Drive)" if info.get("on_drive") else "(EPHEMERAL)")
if prior.is_file() and prior.stat().st_size > 0:
    try:
        import pandas as pd

        ndone = len(pd.read_csv(prior))
        print("RESUME: " + str(ndone) + " completed conditions will be skipped." if ndone
              else "RESUME: empty checkpoint -- starting fresh.")
    except Exception:
        print("RESUME: prior checkpoint unreadable -- starting fresh.")
else:
    print("Fresh run (no prior checkpoint).")

# Belt-and-suspenders: also export the env vars the committed config references.
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["POCKET2MOL_REPO"] = "{P2M_REPO}"
os.environ["POCKET2MOL_ROOT"] = "{P2M_REPO}"
os.environ["POCKET2MOL_CHECKPOINT"] = info["checkpoint"]
os.environ["POCKET2MOL_PYTHON"] = info["p2m_python"]
os.environ["POCKET2MOL_SCRIPT"] = str(ROOT / "scripts" / "pocket2mol_sample_drug_bridge.py")

cmd = [sys.executable, "-u", "-m", "sbdd_robust", "run", "--config", str(cfgp), "--resume"]
# --normalized-only only exists on newer PocketBench; add it if the cloned CLI supports it.
_help = subprocess.run(
    [sys.executable, "-m", "sbdd_robust", "run", "-h"], capture_output=True, text=True
)
if "--normalized-only" in (_help.stdout + _help.stderr):
    cmd.append("--normalized-only")
else:
    print("NOTE: cloned PocketBench CLI has no --normalized-only; running without it "
          "(compute normalized brittleness later via 'pocketbench report' on the CSV).")
print("$ cd", ROOT)
print("$", " ".join(cmd))
proc = subprocess.Popen(
    cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
assert proc.stdout is not None
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
rc = proc.wait()
if rc != 0:
    raise RuntimeError("sbdd_robust exited " + str(rc))

metrics = res_dir / ("metrics_per_condition__run" + run_id + ".csv")
if not metrics.is_file() or metrics.stat().st_size == 0:
    raise RuntimeError(
        "Benchmark finished but metrics CSV missing/empty at " + str(metrics)
        + ". Scroll up for [sbdd_robust] SKIP lines."
    )
import pandas as pd

nrows = len(pd.read_csv(metrics))
print("[benchmark finished] OK --", nrows, "condition rows in", metrics.name)
if nrows < 500:
    print("WARN: expected up to 500 rows (100 pockets x 5 perturbations); only "
          + str(nrows) + ". Re-run this cell to resume remaining conditions.")
"""
        )
    )

    # ---- Cell 10: snapshot ----------------------------------------------------
    c.append(
        md(
            """
### Cell 10 — Snapshot to Drive

If results already live on Drive (checkpointed during the run) this just verifies;
otherwise it mirrors the results dir into `DRIVE_OUT`.
"""
        )
    )
    c.append(
        py(
            """
from __future__ import annotations

import json
import shutil
from pathlib import Path

info = json.loads(Path("/content/sbdd-robust/.pb_paths.json").read_text(encoding="utf-8"))
SRC = Path(info["results"])
if not SRC.is_dir():
    raise RuntimeError("Missing results directory -- Cell 9 must succeed first.")

dst = Path(DRIVE_OUT)
src_res, dst_res = SRC.resolve(), dst.resolve()
already = src_res == dst_res or str(src_res).startswith(str(dst_res) + "/")
if already:
    print("Results already on Drive at", src_res, "-- no copy needed.")
    look_in = SRC
else:
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in sorted(SRC.rglob("*")):
        rel = s.relative_to(SRC)
        t = dst / rel
        if s.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, t)
            n += 1
    print("Copied", n, "files into", dst_res)
    look_in = dst

csvs = list(look_in.glob("metrics_per_condition__*.csv"))
if not csvs:
    raise RuntimeError("metrics_per_condition CSV not found")
print("Example metrics CSV:", csvs[0].name)
"""
        )
    )

    # ---- Cell 11: report ------------------------------------------------------
    c.append(
        md(
            """
### Cell 11 — Reliability report (PBSI + brittleness + crop test)

Runs `pocketbench report` on the Pocket2Mol metrics CSV: normalized brittleness @
τ=0.10, the paired Wilcoxon crop test on ΔQED/ΔSA, and the **Pocket Boundary
Sensitivity Index**. Writes a paper-ready folder to Drive so you can compare PBSI
against DiffSBDD and see whether boundary sensitivity generalizes across architectures.

> If the cloned PocketBench predates the `report` subcommand, this cell **copies the
> metrics CSV to Drive** and tells you to run `pocketbench report` locally on the
> up-to-date repo. The raw per-condition CSV is the deliverable either way.
"""
        )
    )
    c.append(
        py(
            f"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/sbdd-robust")
info = json.loads((ROOT / ".pb_paths.json").read_text(encoding="utf-8"))
run_id = info.get("run_id", "{RUN_ID}")
metrics = Path(info["results"]) / ("metrics_per_condition__run" + run_id + ".csv")
if not metrics.is_file():
    raise RuntimeError("metrics CSV missing: " + str(metrics) + " -- run Cell 9.")

# `report` only exists on newer PocketBench; degrade gracefully if the clone is older.
_help = subprocess.run(
    [sys.executable, "-m", "sbdd_robust", "-h"], capture_output=True, text=True
)
if "report" not in (_help.stdout + _help.stderr):
    dst = Path(DRIVE_OUT) / metrics.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics, dst)
    print("Cloned PocketBench has no 'report' subcommand.")
    print("Saved metrics CSV to:", dst)
    print("Run locally on the up-to-date repo:")
    print("  pocketbench report --metrics <csv> --model pocket2mol "
          "--dataset pocket2mol_real100 --out-dir pocketbench_out")
else:
    out_dir = Path(DRIVE_OUT) / "report"
    cmd = [
        sys.executable, "-m", "sbdd_robust", "report",
        "--metrics", str(metrics),
        "--model", "pocket2mol",
        "--dataset", "pocket2mol_real100",
        "--out-dir", str(out_dir),
    ]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError("report failed:\\n" + (proc.stderr or "")[-6000:])
    print("Report + CSVs written to", out_dir)
"""
        )
    )

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "collapsed_sections": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": c,
    }


def squash_code_cell_sources(nb: dict) -> None:
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
        blob = "".join(cell.get("source") or [])
        try:
            ast.parse(blob, filename=f"cell-{i}")
        except SyntaxError as exc:
            raise SyntaxError(f"Notebook code cell index {i} failed to parse") from exc


def main() -> None:
    out = Path(__file__).resolve().parent / "pocket2mol_real100_benchmark.ipynb"
    nb = build()
    squash_code_cell_sources(nb)
    assert_notebook_code_parses(nb)
    out.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
