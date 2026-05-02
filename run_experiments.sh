#!/bin/bash
set -e

# to start run : chmod +x run_experiments.sh
echo "Starting DoRA pipeline"

# LLaMA-2 7B experiments
echo "LLaMA-2 7B : LoRA baselines"
python code/train.py --method lora --rank 32 --model_id "meta-llama/Llama-2-7b-hf" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_lora_r32_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r32_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r32_adapter

python code/train.py --method lora --rank 64 --model_id "meta-llama/Llama-2-7b-hf" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_lora_r64_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r64_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r64_adapter

echo "LLaMA-2 7B : DoRA variants"
python code/train.py --method dora --rank 16 --model_id "meta-llama/Llama-2-7b-hf" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_dora_r16_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r16_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r16_adapter

python code/train.py --method dora --rank 32 --model_id "meta-llama/Llama-2-7b-hf" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_dora_r32_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r32_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r32_adapter

python code/train.py --method dora --rank 64 --model_id "meta-llama/Llama-2-7b-hf" # to save the adapter weights use -hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Llama-2-7b-hf_dora_r64_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r64_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r64_adapter

# LLaMA-3 8B experiments
echo "LLaMA-3 8B : LoRA baselines"
python code/train.py --method lora --rank 32 --model_id "meta-llama/Meta-Llama-3-8B" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_lora_r32_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r32_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r32_adapter

python code/train.py --method lora --rank 64 --model_id "meta-llama/Meta-Llama-3-8B" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_lora_r64_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r64_final
rm -rf /content/temp_models/Meta-Llama-3-8B_lora_r64_adapter

echo "LLaMA-3 8B : DoRA variants"
python code/train.py --method dora --rank 16 --model_id "meta-llama/Meta-Llama-3-8B" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_dora_r16_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r16_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r16_adapter

python code/train.py --method dora --rank 32 --model_id "meta-llama/Meta-Llama-3-8B" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_dora_r32_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r32_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r32_adapter

python code/train.py --method dora --rank 64 --model_id "meta-llama/Meta-Llama-3-8B" # to save the adapter weights use --hf_user <your_huggingface_username>
python code/evaluate.py --model_path /content/temp_models/Meta-Llama-3-8B_dora_r64_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r64_final
rm -rf /content/temp_models/Meta-Llama-3-8B_dora_r64_adapter

echo "done"