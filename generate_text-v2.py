# text_generator_refactored.py
"""
Generates textual content using a pre-trained single-layer GPT model.

Workflow:
1. Loads the trained language model and its associated tokenizer from a specified directory.
2. Accepts a text prompt from the user or uses a default.
3. Employs the model's generate() method to create new text extending the given prompt.
4. Decodes the generated token sequence and prints it to the console.
5. Computes and displays the perplexity for each generated sequence.
"""

import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# --- Default Configuration Settings ---
INITIAL_MODEL_DIRECTORY = "./models/tinystories_gpt_1layer/final_model" # Default path to the saved model
INITIAL_PROMPT_TEXT = "Once upon a time, in a land far away,"
MAX_OUTPUT_LENGTH = 100  # Max token length for the generated sequence (prompt + generation)
ENABLE_SAMPLING_BY_DEFAULT = True  # Use sampling (True) or greedy decoding (False)
DEFAULT_BEAM_COUNT = 1      # Number of beams for beam search (1 signifies no beam search)
SAMPLING_TEMPERATURE_DEFAULT = 0.7  # Controls randomness: lower is more deterministic
TOP_K_FILTER_DEFAULT = 0       # Considers only the top K tokens for sampling if > 0
NUCLEUS_SAMPLING_P_DEFAULT = 0.95  # Uses nucleus sampling if < 1.0 (considers tokens with cumulative probability > p)
NUM_SEQUENCES_TO_GENERATE_DEFAULT = 1 # Number of distinct sequences to generate

def configure_argument_parser():
    """ Sets up and parses command-line arguments. """
    arg_parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")

    arg_parser.add_argument(
        "--model_dir",
        type=str,
        default=INITIAL_MODEL_DIRECTORY,
        help="Directory path to the trained model (containing weights and tokenizer files)."
    )
    arg_parser.add_argument(
        "--input_prompt",
        type=str,
        default=INITIAL_PROMPT_TEXT,
        help="The initial text prompt for starting text generation."
    )
    arg_parser.add_argument(
        "--max_len",
        type=int,
        default=MAX_OUTPUT_LENGTH,
        help="Maximum token length of the generated text sequence."
    )
    arg_parser.add_argument(
        "--use_sampling",
        type=lambda x: (str(x).lower() == 'true'), # More robust boolean conversion
        default=ENABLE_SAMPLING_BY_DEFAULT,
        help="Enable sampling (True) or use greedy decoding (False)."
    )
    arg_parser.add_argument(
        "--beam_width",
        type=int,
        default=DEFAULT_BEAM_COUNT,
        help="Number of beams for beam search. Set to 1 for no beam search."
    )
    arg_parser.add_argument(
        "--temp",
        type=float,
        default=SAMPLING_TEMPERATURE_DEFAULT,
        help="Sampling temperature. Controls randomness."
    )
    arg_parser.add_argument(
        "--top_k_val",
        type=int,
        default=TOP_K_FILTER_DEFAULT,
        help="Top-K sampling parameter. Filters to K most likely next words."
    )
    arg_parser.add_argument(
        "--top_p_val",
        type=float,
        default=NUCLEUS_SAMPLING_P_DEFAULT,
        help="Nucleus sampling (top-p) parameter. Selects from smallest set of words whose cumulative probability exceeds p."
    )
    arg_parser.add_argument(
        "--num_gen_sequences",
        type=int,
        default=NUM_SEQUENCES_TO_GENERATE_DEFAULT,
        help="Number of different sequences to generate from the prompt."
    )
    return arg_parser.parse_args()

def load_model_and_tokenizer(model_path, current_device):
    """Loads the tokenizer and model from the specified path."""
    print(f"Loading tokenizer and model from: {model_path}")
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
            print(f"Set pad_token to eos_token ({text_tokenizer.pad_token}) as it was not set.")

        language_model = GPT2LMHeadModel.from_pretrained(model_path)
        language_model.to(current_device)  # Move model to the designated device
        language_model.eval()      # Set model to evaluation mode
        print("Model and tokenizer loaded successfully.")
        return text_tokenizer, language_model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure the path is correct and contains all necessary files ")
        print("(e.g., pytorch_model.bin, config.json, tokenizer.json).")
        print("These files are typically saved by a training script (e.g., train_gpt.py) in a 'final_model' subdirectory.")
        return None, None

def generate_text_sequences(language_model, text_tokenizer, encoded_prompt, cli_arguments):
    """Generates text sequences using the model."""
    print(f"\nGenerating text based on prompt: '{cli_arguments.input_prompt}'")
    print("Generation Parameters:")
    print(f"  Max Length: {cli_arguments.max_len}")
    print(f"  Use Sampling: {cli_arguments.use_sampling}")
    print(f"  Beam Width: {cli_arguments.beam_width}")
    print(f"  Temperature: {cli_arguments.temp}")
    print(f"  Top-K: {cli_arguments.top_k_val}")
    print(f"  Top-P: {cli_arguments.top_p_val}")
    print(f"  Number of Sequences: {cli_arguments.num_gen_sequences}\n")

    with torch.no_grad():  # Disable gradient calculations during inference
        generated_token_sequences = language_model.generate(
            encoded_prompt,
            max_length=cli_arguments.max_len,
            do_sample=cli_arguments.use_sampling,
            num_beams=cli_arguments.beam_width,
            temperature=cli_arguments.temp,
            top_k=cli_arguments.top_k_val,
            top_p=cli_arguments.top_p_val,
            num_return_sequences=cli_arguments.num_gen_sequences,
            pad_token_id=text_tokenizer.eos_token_id, # Crucial for proper sequence termination
            no_repeat_ngram_size=2  # Optional: helps prevent repetitive n-grams
        )
    return generated_token_sequences

def main():
    """ Main execution function: loads model, prepares input, and generates text. """
    cli_arguments = configure_argument_parser()

    # Determine computation device (GPU if available, else CPU)
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {computation_device}")

    # 1. Load Tokenizer and Model
    text_tokenizer, language_model = load_model_and_tokenizer(cli_arguments.model_dir, computation_device)
    if not text_tokenizer or not language_model:
        return # Exit if loading failed

    # 2. Prepare Input Prompt
    # Encode the prompt text into token IDs, ensuring output is a PyTorch tensor
    encoded_prompt_tokens = text_tokenizer.encode(cli_arguments.input_prompt, return_tensors="pt").to(computation_device)

    # 3. Generate Text
    generated_token_sequences = generate_text_sequences(language_model, text_tokenizer, encoded_prompt_tokens, cli_arguments)

    # 4. Decode and Print Output, and Calculate Perplexity
    print("--- Generated Text ---")
    for i, single_sequence_tokens in enumerate(generated_token_sequences):
        decoded_output_text = text_tokenizer.decode(single_sequence_tokens, skip_special_tokens=True)
        print(f"\nSequence {i + 1}:")
        print(decoded_output_text)
        print()

        # Compute perplexity for the generated sequence
        # Ensure the sequence has a batch dimension for the model
        # Note: Perplexity calculation here is on the generated output itself, which might not be standard.
        # Usually, perplexity is calculated on a held-out test set using the model's probabilities for the true next tokens.
        # This calculation provides a measure of the model's confidence in its own generation.
        try:
            # We need to pass both input_ids and labels. For perplexity of the generated sequence,
            # labels are the same as input_ids (shifted internally by the model for causal LM).
            model_input_ids = single_sequence_tokens.unsqueeze(0) # Add batch dimension: [1, seq_len]
            
            # For calculating perplexity of just the generated part, one might consider
            # using only the generated tokens as labels and the full sequence (prompt + generation) as input_ids.
            # However, for simplicity here, we use the whole generated sequence.
            loss = language_model(input_ids=model_input_ids, labels=model_input_ids).loss
            sequence_perplexity = torch.exp(loss)
            print(f"Perplexity: {sequence_perplexity.item():.2f}")
        except Exception as e:
            print(f"Could not compute perplexity: {e}")
            
        print("-" * 20) # Separator

if __name__ == "__main__":
    main()