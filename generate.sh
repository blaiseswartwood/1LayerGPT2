#!/bin/bash

source ./venv_tinystories/bin/activate
python generate_text.py \
--model_path ./models/tinystories_gpt_1layer/final_model \
--prompt "In the dark of night, the moon shone brightly on" \
--max_length 100 \
--temperature 0.7 \
--top_k 50 \
--num_return_sequences 3

