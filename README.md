# dora-repro
A PyTorch Reimplementation of Weight-Decomposed Low-Rank Adaptation (DoRA)

## Load saved DoRA adapters

If you pushed a DoRA adapter repo (with `dora_adapter.bin` + `adapter_config.json`), rebuild a local model with:

```bash
python code/load_dora_adapter.py \
  --adapter_source <hf_user>/<repo_name> \
  --output_dir /content/temp_models/restored_dora \
  --merge
```
