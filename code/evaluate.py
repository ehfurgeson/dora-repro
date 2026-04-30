import argparse
import subprocess

def evaluate_model(model_path, 
                   tasks="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"):
    print(f"evalulating model at {model_path}")

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks,
        "--device", "cuda:0",
        "--batch_size", "8"
    ]
    
    result = subprocess.run(cmd, capture_output = True, text = True)

    clean_name = model_path.split("/")[-1]
    log_name = f"{clean_name}_eval.txt"
    with open(f"./results/{log_name}", "w") as f:
        f.write(result.stdout)

    print(f"eval saved at {log_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type = str, required = True)
    args = parser.parse_args()
    evaluate_model(args.model_path)