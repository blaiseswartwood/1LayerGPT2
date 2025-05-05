#!/bin/bash

source ./venv_tinystories/bin/activate
python3 preprocess_tinystories.py \
--output_dir ./data/tokenized_tinystories \
--num_proc 64 
