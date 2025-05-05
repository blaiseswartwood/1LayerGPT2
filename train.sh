#!/bin/bash

source ./venv_tinystories/bin/activate
python train_gpt.py \
--output_dir ./models/tinystories_gpt_1layer \
--tokenized_data_path ./data/tokenized_tinystories \
--gpus 8
--batch_size 8
--accumulate_grad_batches 4 
--num_epochs 3
--learning_rate 5e-5
--precision 16

