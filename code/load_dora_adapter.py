import argparse
import json
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from dora import apply_dora, load_dora_adapter_state, merge_and_unload_dora

def resolve_adapter_path(adapter_source):
    if os.path.isdir(adapter_source):
        return adapter_source
    return snapshot_download(repo_id = adapter_source)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_source", type = str, required = True,
                        help = "Local adapter folder or HF repo id (user/repo)")
    parser.add_argument("--output_dir", type = str, required = True,
                        help = "Directory to save merged model")
    parser.add_argument("--merge", action = "store_true",
                        help = "Merge DoRA adapter into base model weights before saving")
    args = parser.parse_args()

    adapter_dir = resolve_adapter_path(args.adapter_source)
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "dora_adapter.bin")

    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    base_model_id = adapter_config["base_model"]
    rank = adapter_config["rank"]

    print(f"loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        attn_implementation = "sdpa"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    for param in model.parameters():
        param.requires_grad = False
    model = apply_dora(model, rank = rank)

    adapter_state = torch.load(weights_path, map_location = "cpu")
    model = load_dora_adapter_state(model, adapter_state)

    if args.merge:
        model = merge_and_unload_dora(model)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model to {args.output_dir}")

if __name__ == "__main__":
    main()
