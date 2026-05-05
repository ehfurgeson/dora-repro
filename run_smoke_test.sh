#!/bin/bash
set -e

# to start run : chmod +x run_smoke_test.sh
# End-to-end smoke test: exercises the full train.py + evaluate.py
# pipeline at tiny scale (5 training steps, 10 BoolQ examples) for both
# LoRA and DoRA paths. Intended to be run BEFORE the long sweep to
# catch HF auth, data path, OOM, lm_eval install, and merge-bug issues.
#
# Uses rank=2 so smoke-test artifacts can't collide with any production
# r=4/8/16/32/64 results. Cleans up temp_models/ and results/ on exit.
# Expected wall time: ~5-10 minutes on A100 (mostly model download/load).

echo "=== DoRA smoke test ==="

# Fail fast if the training data isn't where data_utils.py expects it.
if [ ! -f "data/ft-training-set/commonsense_170k.json" ]; then
  echo "ERROR: data/ft-training-set/commonsense_170k.json not found."
  echo "Make sure the working dir is the project root and the dataset is mounted."
  exit 1
fi

# --- LoRA path (PEFT) ---
echo ""
echo "[1/2] LoRA r=2 smoke ..."
python code/train.py --method lora --rank 2 \
  --model_id "meta-llama/Llama-2-7b-hf" --max_steps 5
python code/evaluate.py \
  --model_path /content/temp_models/Llama-2-7b-hf_lora_r2_final \
  --tasks boolq --limit 10
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r2_final
rm -rf /content/temp_models/Llama-2-7b-hf_lora_r2_adapter

# --- DoRA path (custom DoRALayer + merge_and_unload) ---
echo ""
echo "[2/2] DoRA r=2 smoke ..."
python code/train.py --method dora --rank 2 \
  --model_id "meta-llama/Llama-2-7b-hf" --max_steps 5
python code/evaluate.py \
  --model_path /content/temp_models/Llama-2-7b-hf_dora_r2_final \
  --tasks boolq --limit 10
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r2_final
rm -rf /content/temp_models/Llama-2-7b-hf_dora_r2_adapter

# --- verify both eval logs landed and contain a results table ---
echo ""
echo "Verifying smoke-test eval outputs ..."
for f in results/Llama-2-7b-hf_lora_r2_final_eval.txt \
         results/Llama-2-7b-hf_dora_r2_final_eval.txt; do
  if [ ! -s "$f" ]; then
    echo "FAIL: $f is missing or empty"
    exit 1
  fi
  if ! grep -q "boolq" "$f"; then
    echo "FAIL: $f did not contain a boolq result row"
    cat "$f"
    exit 1
  fi
  echo "OK: $f"
done

# clean up smoke-test result files so they don't pollute results/
#rm -f results/Llama-2-7b-hf_lora_r2_final_eval.txt
#rm -f results/Llama-2-7b-hf_dora_r2_final_eval.txt

echo ""
echo "=== smoke test PASSED ==="
echo "safe to run run_low_rank_llama2.sh / run_low_rank_llama3.sh"
