#!/bin/bash
set -e

# to start run : chmod +x run_low_rank_llama2.sh
# Low-rank sweep on Llama-2-7B to enable the rank-robustness analysis
# (DoRA paper Fig. 4). Adds LoRA r=4/8/16 and DoRA r=4/8 to complement
# the existing r=16/32/64 runs in run_experiments.sh.
# Expected wall time on a single A100: ~5-7.5 hours.

echo "Starting DoRA low-rank pipeline (Llama-2-7B)"

# LoRA low-rank baselines
echo "LLaMA-2 7B : LoRA r=4"
python code/train.py --method lora --rank 4 --model_id "meta-llama/Llama-2-7b-hf"
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_lora_r4_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r4_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r4_adapter

echo "LLaMA-2 7B : LoRA r=8"
python code/train.py --method lora --rank 8 --model_id "meta-llama/Llama-2-7b-hf"
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_lora_r8_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r8_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r8_adapter

echo "LLaMA-2 7B : LoRA r=16"
python code/train.py --method lora --rank 16 --model_id "meta-llama/Llama-2-7b-hf"
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_lora_r16_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r16_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r16_adapter

# DoRA low-rank variants
echo "LLaMA-2 7B : DoRA r=4"
python code/train.py --method dora --rank 4 --model_id "meta-llama/Llama-2-7b-hf"
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_dora_r4_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r4_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r4_adapter

echo "LLaMA-2 7B : DoRA r=8"
python code/train.py --method dora --rank 8 --model_id "meta-llama/Llama-2-7b-hf"
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_dora_r8_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r8_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r8_adapter

echo "done (Llama-2-7B low-rank sweep)"
