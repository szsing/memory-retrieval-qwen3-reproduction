# Haiku cache replacement diagnostic

This runner was executed on the existing server environment. It uses the source snapshot and strict format adapter in `experiments/original-qwen-format-v2/`.

The four primary groups replace only memory/request property lists: QQ, HQ, QH, HH. Three additional controls use raw requests, property presence, or shuffled request properties. All groups freeze Qwen HyDE and gate outputs and use the same Qwen picker, embeddings, PV2 prompts, weight 8, and wide candidate rule. The job completed; no unrelated paused benchmark was resumed.

The runner requires the separately supplied `haiku-cache.json` plus original benchmark data and model service. Full input/cache files and model weights are intentionally not published. This is not a standalone one-command reproduction from public files. The 7200-second job window is specific to this diagnostic; review it before any new run. Private cache filenames in the runner refer to the server working directory, not artifacts provided in this repository.

Published results are under `results/haiku-swap-20261123-v1/`. Recompute the public records without models:

```bash
python3 scripts/verify_haiku_swap.py
```

See `docs/HAIKU_SWAP_20261123.md` and `docs/HAIKU_PICKER_AUDIT.md` for metric definitions, publication boundaries, and conclusions. This diagnostic did not rerun RootMem and does not establish superiority over it.
