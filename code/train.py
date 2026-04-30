import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling as dcflm
from peft import LoraConfig, get_peft_model
from data_utils import load_commonsense
from dora import apply_dora, merge_and_unload_dora

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type = str, choices = ["lora", "dora"], required = True)
    parser.add_argument("--rank", type = int, required = True)
    parser.add_argument("--model_id", type = str, default = "meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_user", type = str, required = False)
    args = parser.parse_args()

    model_id = args.model_id 

    print(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = torch.bfloat16,
        device_map = "auto" 
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # applied standard LoRA
    if args.method == "lora":
        config = LoraConfig(
            r = args.rank,
            lora_alpha = args.rank * 2,
            target_modules = ["q_proj", "v_proj"],
            bias = "none",
            task_type = "CAUSAL_LM"
        )
        model = get_peft_model(model, config)
    elif args.method == "dora":
        for param in model.parameters(): 
            param.requires_grad = False
        model = apply_dora(model, rank = args.rank)

    train_data = load_commonsense(tokenizer)

    if "train" in train_data:
        train_data = train_data["train"]
    
    clean_model_name = model_id.split("/")[-1]
    save_dir = f"/content/temp_models/{clean_model_name}_{args.method}_r{args.rank}"

    training_args = TrainingArguments(
        output_dir = save_dir,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,
        num_train_epochs = 3,
        lr_scheduler_type = "cosine",
        warmup_steps = 50,
        logging_steps = 10,
        save_strategy = "epoch",
        bf16 = True,
    )

    trainer = Trainer(
        model = model,
        train_dataset = train_data,
        args = training_args,
		data_collator = dcflm(tokenizer, mlm = False),
    )

    print(f"training {args.method} rank {args.rank}")
    trainer.train()

    if args.method == "dora":
        trainer.model = merge_and_unload_dora(trainer.model)
    elif args.method == "lora":
        trainer.model = trainer.model.merge_and_unload()

    trainer.model.save_pretrained(f"{save_dir}_final")
    tokenizer.save_pretrained(f"{save_dir}_final")

    # optionally saving finetuned models to hf
    # if you do this make sure you have a write token to hf
    if args.hf_user:
        repo_name = f"{args.hf_user}/{clean_model_name}-{args.method}-r{args.rank}"
        print(f"Pushing model to Hugging Face Hub: {repo_name}...")
        trainer.model.push_to_hub(repo_name, private = True)
        tokenizer.push_to_hub(repo_name, private = True)


if __name__ == "__main__":
    main()