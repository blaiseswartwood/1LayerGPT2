#!/bin/bash

# Activate the virtual environment
source ./venv_tinystories/bin/activate

# --- Execution Commands with Updated Arguments ---

# Example 1: Greedy decoding (use_sampling=False)
echo "Running with greedy decoding..."
python generate_text-v2.py \
  --model_dir ./models/tinystories_gpt_1layer/final_model \
  --input_prompt "Once upon a time, in a land far away," \
  --max_len 100 \
  --use_sampling False \
  --beam_width 1 \
  --temp 1.0 \
  --top_k_val 0 \
  --top_p_val 1 \
  --num_gen_sequences 1

echo -e "\n---------------------------------------------\n"

# Example 2: Sampling with temperature 1.0
echo "Running with sampling, temperature 1.0..."
python generate_text-v2.py \
  --model_dir ./models/tinystories_gpt_1layer/final_model \
  --input_prompt "Once upon a time, in a land far away," \
  --max_len 100 \
  --use_sampling True \
  --beam_width 1 \
  --temp 1.0 \
  --top_k_val 0 \
  --top_p_val 1 \
  --num_gen_sequences 1

echo -e "\n---------------------------------------------\n"

# Example 3: Sampling with temperature 2.0 (higher temperature for more randomness)
echo "Running with sampling, temperature 2.0..."
python generate_text-v2.py \
  --model_dir ./models/tinystories_gpt_1layer/final_model \
  --input_prompt "Once upon a time, in a land far away," \
  --max_len 100 \
  --use_sampling True \
  --beam_width 1 \
  --temp 2.0 \
  --top_k_val 0 \
  --top_p_val 1 \
  --num_gen_sequences 1

# Deactivate the virtual environment (optional, good practice)
# deactivate
