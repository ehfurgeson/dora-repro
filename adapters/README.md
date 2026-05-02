# Using Saved LoRA/DoRA Adapters

This project stores adapter-only checkpoints (not full finetuned model weights).  
You can keep all adapters in Google Drive and rebuild/evaluate models from them.

## Prerequisite: gated base model access

These adapters were trained on gated Llama base models (`meta-llama/Llama-2-7b-hf` and `meta-llama/Meta-Llama-3-8B`), so you must first request and receive access to those models on Hugging Face with the same account you use in Colab/local runs.
After access is approved, authenticate in your runtime (for example with `huggingface-cli login` or `notebook_login`) using a token that can read gated repos.

## 1) Suggested folder layout

```text
MyDrive/
  dora-repro/
    code/
    run_experiments.sh
    adapters/
      Llama2-7B-LoRA-r32/
      Llama2-7B-LoRA-r64/
      Llama2-7B-DoRA-r16/
      Llama2-7B-DoRA-r32/
      Llama2-7B-DoRA-r64/
      Llama3-8B-LoRA-r32/
      Llama3-8B-LoRA-r64/
      Llama3-8B-DoRA-r16/
      Llama3-8B-DoRA-r32/
      Llama3-8B-DoRA-r64/
```

Each adapter folder should contain what was saved from training:
- **LoRA** adapters: PEFT adapter files (`adapter_config.json`, `adapter_model.safetensors`/`.bin`, tokenizer files, etc.)
- **DoRA** adapters (custom in this repo):  
  - `adapter_config.json`  
  - `dora_adapter.bin`  
  - tokenizer files (`tokenizer.json`, etc.)

---

## 2) Colab setup (with Google Drive)

```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/dora-repro
```

---

## 3) Evaluate a LoRA adapter

For LoRA adapters, load base model + adapter using PEFT, then evaluate.

Example Python snippet:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-2-7b-hf"
adapter_path = "/content/drive/MyDrive/dora-repro/adapters/Llama2-7B-LoRA-r32"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model = PeftModel.from_pretrained(model, adapter_path)

# optional: merge to plain HF model
model = model.merge_and_unload()
model.save_pretrained("/content/temp_models/lora_r32_merged")
tokenizer.save_pretrained("/content/temp_models/lora_r32_merged")
```

Then run:
```bash
python code/evaluate.py --model_path /content/temp_models/lora_r32_merged
```

---

## 4) Evaluate a DoRA adapter (custom)

Use the helper script already in this repo:

```bash
python code/load_dora_adapter.py \
  --adapter_source /content/drive/MyDrive/dora-repro/adapters/Llama2-7B-DoRA-r32 \
  --output_dir /content/temp_models/dora_r32_merged \
  --merge
```

Then run:
```bash
python code/evaluate.py --model_path /content/temp_models/dora_r32_merged
```

> `--merge` is recommended for compatibility with current evaluation flow.

---

## 5) Loading directly from Hugging Face instead of Drive

If an adapter is on HF, set `--adapter_source` to the repo id:

```bash
python code/load_dora_adapter.py \
  --adapter_source ehfurgeson/Llama2-7B-DoRA-r32 \
  --output_dir /content/temp_models/dora_r32_merged \
  --merge
```

---

## 6) Naming convention recommendations

Use a consistent pattern to avoid confusion:

`<ModelFamily>-<Method>-r<Rank>`

Examples:
- `Llama2-7B-LoRA-r32`
- `Llama2-7B-DoRA-r64`
- `Llama3-8B-LoRA-r64`
- `Llama3-8B-DoRA-r16`

---

## 7) Common gotchas

- **Base model mismatch**: adapter must be loaded on the exact base model family/version used in training.
- **Tokenizer mismatch**: use tokenizer saved with adapter folder when possible.
- **OOM issues**: if needed, use fewer GPUs/lower batch size during evaluation.
- **Folder typo**: use `adapters/` (not `adpaters/`) to keep scripts clean.