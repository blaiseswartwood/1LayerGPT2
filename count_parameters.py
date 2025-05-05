# generate_text.py
"""
Generates text using the trained one-layer GPT model.

Steps:
1. Loads the trained model and tokenizer from the specified directory.
2. Takes a text prompt as input.
3. Uses the model's generate() method to produce new text based on the prompt.
4. Decodes and prints the generated text.
"""

import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# --- Configuration ---
DEFAULT_MODEL_PATH = "./models/tinystories_gpt_1layer/final_model" # Path where train_gpt.py saved the final model

def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model directory (containing model weights and tokenizer).")
    return parser.parse_args()

def main():
    """ Main function to load model and generate text. """
    args = parse_args()

    # Determine device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # Ensure pad token is set correctly after loading
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
             print(f"Set pad_token to eos_token ({tokenizer.pad_token}) after loading.")

        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.to(device) # Move model to the selected device
        model.eval()     # Set model to evaluation mode (disables dropout, etc.)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Ensure the path is correct and contains the necessary files ")
        print("(pytorch_model.bin, config.json, tokenizer.json, etc.)")
        print("These should be saved by train_gpt.py in the 'final_model' subdirectory.")
        return

    print("Model and tokenizer loaded successfully.")

    # 2. Count the parameters
    print(model.config.to_json_string())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Parameters:")
    print("Total trainable parameters: ", total_params, "\n")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, '-->', param.data.size())

if __name__ == "__main__":
    main()
