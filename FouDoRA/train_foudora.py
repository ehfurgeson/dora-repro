import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling as dcflm
from data_utils import load_commonsense
from FouDoRA import apply_foudora, merge_and_unload_foudora


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM with FouDoRA (Fourier-based DoRA)."
    )
    parser.add_argument(
        "--n_freqs",
        type=int,
        default=8,
        help="Number of low-frequency bins to adapt per FFT dimension (analogous to rank).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="Scaling factor for the spectral delta (alpha / n_freqs is applied).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Which Linear sub-layers to wrap with FouDoRA.",
    )
    parser.add_argument(
        "--hf_user",
        type=str,
        required=False,
        help="HuggingFace username; if set, the merged model is pushed to the Hub.",
    )
    args = parser.parse_args()

    print(f"Loading {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Freeze all base parameters, then wrap target layers with FouDoRA.
    for param in model.parameters():
        param.requires_grad = False

    model = apply_foudora(
        model,
        n_freqs=args.n_freqs,
        alpha=args.alpha,
        target_modules=args.target_modules,
    )

    n_trainable = count_trainable_params(model)
    print(f"Trainable parameters: {n_trainable:,}")

    train_data = load_commonsense(tokenizer)
    if "train" in train_data:
        train_data = train_data["train"]

    clean_model_name = args.model_id.split("/")[-1]
    save_dir = f"/content/temp_models/{clean_model_name}_foudora_f{args.n_freqs}"

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=1000,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=dcflm(tokenizer, mlm=False),
    )

    print(f"Training FouDoRA  n_freqs={args.n_freqs}  alpha={args.alpha}")
    trainer.train()

    trainer.model = merge_and_unload_foudora(trainer.model)

    final_dir = f"{save_dir}_final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Merged model saved to {final_dir}")

    if args.hf_user:
        repo_name = f"{args.hf_user}/{clean_model_name}-foudora-f{args.n_freqs}"
        print(f"Pushing to Hub: {repo_name}")
        trainer.model.push_to_hub(repo_name, private=True)
        tokenizer.push_to_hub(repo_name, private=True)


if __name__ == "__main__":
    main()
