# Configuration layout

- **`examples/`** — small configs for learning the tool (`smoke.yaml`, `diffsbdd.example.yaml`, `real5*.yaml`, …). Start here.
- **`experiments/`** — full paper-scale runs (47-pocket invariant panel, meaningful mutations, n=50 subset, …). See `experiments/README.md`.

Run the toolkit with:

```bash
python -m sbdd_robust run --config configs/examples/smoke.yaml
```

Use **`configs/experiments/`** only when reproducing the manuscript or running large benchmarks.
