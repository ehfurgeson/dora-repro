# to start run : chmod +x run_experiments.sh
echo "starting DoRA pipeline"

# LoRA baselines
python code/train.py --method lora --rank 32
python code/train.py --method lora --rank 64

# DoRA variants
python code/train.py --method dora --rank 16
python code/train.py --method dora --rank 32
python code/train.py --method dora --rank 64

# evaluating it all
echo "evaluating"
python code/evaluate.py --model_path ./results/lora_r32_final
python code/evaluate.py --model_path ./results/lora_r64_final
python code/evaluate.py --model_path ./results/dora_r16_final
python code/evaluate.py --model_path ./results/dora_r32_final
python code/evaluate.py --model_path ./results/dora_r64_final

echo "done"