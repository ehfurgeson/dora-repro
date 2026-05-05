#!/bin/bash
set -e

# to start run : chmod +x run_low_rank_llama3.sh
# Low-rank sweep on Llama-3-8B to enable the rank-robustness analysis
# (DoRA paper Fig. 4). Adds LoRA r=4/8/16 and DoRA r=4/8 to complement
# the existing r=16/32/64 runs in run_experiments.sh.
# Expected wall time on a single A100: ~6-9 hours (Llama-3 is slightly
# larger than Llama-2 and uses smaller eval batches in the existing logs).

echo "Starting DoRA low-rank pipeline (Llama-3-8B)"

# LoRA low-rank baselines
echo "LLaMA-3 8B : LoRA r=4"
python code/train.py --method lora --rank 4 --model_id "meta-llama/Meta-Llama-3-8B"
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_lora_r4_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r4_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r4_adapter

echo "LLaMA-3 8B : LoRA r=8"
python code/train.py --method lora --rank 8 --model_id "meta-llama/Meta-Llama-3-8B"
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_lora_r8_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r8_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r8_adapter

echo "LLaMA-3 8B : LoRA r=16"
python code/train.py --method lora --rank 16 --model_id "meta-llama/Meta-Llama-3-8B"
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_lora_r16_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r16_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r16_adapter

# DoRA low-rank variants
echo "LLaMA-3 8B : DoRA r=4"
python code/train.py --method dora --rank 4 --model_id "meta-llama/Meta-Llama-3-8B"
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_dora_r4_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r4_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r4_adapter

echo "LLaMA-3 8B : DoRA r=8"
python code/train.py --method dora --rank 8 --model_id "meta-llama/Meta-Llama-3-8B"
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_dora_r8_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r8_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r8_adapter

echo "done (Llama-3-8B low-rank sweep)"
