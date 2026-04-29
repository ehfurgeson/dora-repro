from datasets import load_dataset

def load_commonsense(tokenizer, max_length = 512):
    dataset = load_dataset("json", data_files={"train": "data/ft-training-set/commonsense_170k.json"})

    def format_prompt(example):
        instruction = example["instructions"]
        output = example["output"]

        prompt = f"{instruction}\n{output}"

        tokens = tokenizer(
            prompt,
            truncation = True,
            max_length = max_length,
            padding = "max_length"
        )

        tokens["labels"] = tokens["input_ids"].copy()

        return tokens
    
    columns_to_remove = ["instruction", "input", "output", "answer"]
    tokenized_dataset = dataset.map(format_prompt, 
                                    remove_columns = columns_to_remove)
    
    return tokenized_dataset
