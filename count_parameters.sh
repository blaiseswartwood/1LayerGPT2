#!/bin/bash

source ./venv_tinystories/bin/activate
python count_parameters.py \
--model_path ./models/tinystories_gpt_1layer/final_model

