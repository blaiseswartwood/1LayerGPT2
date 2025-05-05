# preprocess_tinystories.py
"""
Preprocesses the TinyStories dataset for training a GPT model.

Steps:
1. Loads the TinyStories dataset from Hugging Face Hub.
2. Loads a pre-trained tokenizer (e.g., GPT-2).
3. Tokenizes the dataset texts.
4. Chunks the tokenized texts into fixed-length sequences.
5. Saves the processed dataset to disk for efficient loading during training.

This script leverages multiple CPU cores for faster processing via `datasets.map()`.
"""

import os
import argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging
from concurrent.futures import ProcessPoolExecutor # Used indirectly by datasets.map

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Sensible defaults, can be overridden by command-line arguments
DEFAULT_DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_TOKENIZER_NAME = "gpt2" # Using a standard tokenizer
DEFAULT_OUTPUT_DIR = "./data/tokenized_tinystories"
DEFAULT_NUM_PROC = os.cpu_count() # Use all available CPU cores
DEFAULT_BLOCK_SIZE = 512 # Sequence length for the model

def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Preprocess TinyStories dataset.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER_NAME,
                        help="Name of the pre-trained tokenizer on Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the tokenized dataset.")
    parser.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC,
                        help="Number of CPU cores to use for tokenization.")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="The maximum sequence length for model inputs.")
    return parser.parse_args()

def tokenize_function(examples, tokenizer):
    """
    Tokenizes text data. It processes a batch of examples.
    We concatenate all texts first and then tokenize, which is generally efficient.
    """
    # The tokenizer handles padding and truncation, but here we focus on just tokenizing.
    # We will handle the chunking into block_size later.
    # We assume the dataset has a 'text' column.
    return tokenizer(examples["text"])

def group_texts(examples, block_size):
    """
    Groups tokenized texts into blocks of a specified size (block_size).
    This is crucial for creating fixed-length sequences for the language model.
    """
    # Concatenate all texts in the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Drop the last partial block to ensure all blocks are of size block_size
    total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Create 'labels' for language modeling (input_ids shifted) - the model handles this internally
    # during training if labels aren't provided, but it's common practice.
    # However, for simplicity with GPT2LMHeadModel, we often just pass input_ids,
    # and the model creates labels by shifting internally. Let's keep it simple here.
    # result["labels"] = result["input_ids"].copy() # Optional: Can be added if needed by specific training setup
    return result

def main():
    """ Main function to execute the preprocessing pipeline. """
    args = parse_args()
    logging.info(f"Starting preprocessing with args: {args}")

    # 1. Load Tokenizer
    logging.info(f"Loading tokenizer: {args.tokenizer_name}")
    # Using trust_remote_code=True might be necessary for some tokenizers/models
    # but generally avoided unless the source is known and trusted. GPT-2 is standard.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Set padding token if it doesn't exist (GPT-2 uses EOS token for padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Tokenizer pad_token set to eos_token ({tokenizer.pad_token})")

    # 2. Load Dataset
    logging.info(f"Loading dataset: {args.dataset_name}")
    # TinyStories usually has 'train' and 'validation' splits
    raw_datasets = load_dataset(args.dataset_name)
    logging.info(f"Dataset loaded: {raw_datasets}")

    # Ensure 'text' column exists
    # Example: Check the first example of the training set
    if 'train' in raw_datasets and len(raw_datasets['train']) > 0:
        first_example = raw_datasets['train'][0]
        if 'text' not in first_example:
            raise ValueError("Dataset split 'train' does not contain a 'text' column.")
        logging.info("Dataset structure seems correct (contains 'text' column).")
    else:
        logging.warning("Could not verify 'text' column in 'train' split (split missing or empty).")


    # 3. Tokenize Dataset
    logging.info(f"Tokenizing dataset using {args.num_proc} processes...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,          # Process data in batches
        num_proc=args.num_proc, # Use multiple cores
        remove_columns=raw_datasets["train"].column_names # Remove original text column
    )
    logging.info(f"Tokenization complete: {tokenized_datasets}")

    # 4. Group into Blocks (Chunking)
    logging.info(f"Grouping texts into blocks of size {args.block_size}...")
    processed_datasets = tokenized_datasets.map(
        group_texts,
        fn_kwargs={'block_size': args.block_size},
        batched=True,          # Process data in batches
        num_proc=args.num_proc  # Use multiple cores
    )
    logging.info(f"Grouping complete: {processed_datasets}")
    logging.info(f"Example processed entry keys: {processed_datasets['train'][0].keys()}")
    logging.info(f"Number of samples - Train: {len(processed_datasets['train'])}, Validation: {len(processed_datasets['validation'])}")


    # 5. Save Processed Dataset
    logging.info(f"Saving processed dataset to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_datasets.save_to_disk(args.output_dir)
    logging.info("Preprocessing finished successfully!")

if __name__ == "__main__":
    main()
