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
DEFAULT_PROMPT = "Once upon a time, in a land far away,"
DEFAULT_MAX_LENGTH = 100 # Maximum length of the generated sequence (including prompt)
DEFAULT_TEMPERATURE = 0.7 # Controls randomness (lower = more deterministic)
DEFAULT_TOP_K = 50       # Considers only the top K tokens for sampling
DEFAULT_TOP_P = 0.95     # Uses nucleus sampling (considers tokens cumulative prob > p)
DEFAULT_NUM_RETURN_SEQUENCES = 1 # How many different sequences to generate

def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model directory (containing model weights and tokenizer).")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="The starting prompt for text generation.")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Maximum length of the generated text sequence.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Top-K sampling parameter.")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                        help="Nucleus sampling (top-p) parameter.")
    parser.add_argument("--num_return_sequences", type=int, default=DEFAULT_NUM_RETURN_SEQUENCES,
                        help="Number of sequences to generate.")
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

    # 2. Prepare Input Prompt
    # Encode the prompt text into token IDs
    inputs = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    # `return_tensors="pt"` ensures output is a PyTorch tensor

    # 3. Generate Text
    print(f"\nGenerating text based on prompt: '{args.prompt}'")
    print("Generation parameters:")
    print(f"  max_length: {args.max_length}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_k: {args.top_k}")
    print(f"  top_p: {args.top_p}")
    print(f"  num_return_sequences: {args.num_return_sequences}\n")

    # Use the model's generate method
    # `no_repeat_ngram_size` can prevent repetitive phrases
    # `do_sample=True` is required for temperature, top_k, top_p to have an effect
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True, # Enable sampling
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.eos_token_id # Important for generation completion
            # no_repeat_ngram_size=2 # Optional: prevent repeating 2-grams
        )

    # 4. Decode and Print Output
    print("--- Generated Text ---")
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\nSequence {i+1}:")
        print(generated_text)
        print("-" * 20)

if __name__ == "__main__":
    main()
