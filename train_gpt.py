# train_gpt.py
"""
Trains a one-layer GPT model on the preprocessed TinyStories dataset using PyTorch Lightning.

Steps:
1. Defines a PyTorch Lightning Module (`LitGPT`) containing the GPT model,
   training logic (loss calculation, optimizer configuration).
2. Defines a PyTorch Lightning DataModule (`TinyStoriesDataModule`) to load
   the preprocessed data efficiently.
3. Configures and runs the PyTorch Lightning Trainer for distributed training
   across multiple GPUs.
4. Saves model checkpoints during training.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
#from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


# --- Configuration ---
DEFAULT_TOKENIZED_DATA_PATH = "./data/tokenized_tinystories"
DEFAULT_TOKENIZER_NAME = "gpt2" # Must match the one used in preprocessing
DEFAULT_OUTPUT_DIR = "./models/tinystories_gpt_1layer"
DEFAULT_MODEL_NAME = "tinystories_gpt_1layer"

# Model Hyperparameters (Students can experiment here!)
DEFAULT_N_LAYER = 1       # Core requirement: one layer
DEFAULT_N_EMBD = 768     # Embedding dimension (e.g., 768 for GPT-2 base)
DEFAULT_N_HEAD = 12      # Number of attention heads (e.g., 12 for GPT-2 base)
DEFAULT_BLOCK_SIZE = 512 # Sequence length (must match preprocessing)

# Training Hyperparameters (Students can experiment here!)
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 8     # Adjust based on GPU memory (per GPU)
DEFAULT_ACCUMULATE_GRAD_BATCHES = 4 # Effective batch size = BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * NUM_GPUS
DEFAULT_NUM_EPOCHS = 3
DEFAULT_WARMUP_STEPS = 100
DEFAULT_GPUS = torch.cuda.device_count() # Use all available GPUs
DEFAULT_PRECISION = 16 # Use mixed precision (16-bit) for speed and memory savings
DEFAULT_NUM_WORKERS = 4 # Dataloader workers per GPU

def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Train a one-layer GPT on TinyStories.")
    # Data/Model Paths
    parser.add_argument("--tokenized_data_path", type=str, default=DEFAULT_TOKENIZED_DATA_PATH)
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    # Model Hyperparameters
    parser.add_argument("--n_layer", type=int, default=DEFAULT_N_LAYER, help="Number of transformer layers (fixed to 1).")
    parser.add_argument("--n_embd", type=int, default=DEFAULT_N_EMBD, help="Embedding dimension.")
    parser.add_argument("--n_head", type=int, default=DEFAULT_N_HEAD, help="Number of attention heads.")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="Sequence length.")
    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per GPU.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=DEFAULT_ACCUMULATE_GRAD_BATCHES)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS, help="Number of GPUs to use.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Training precision (16 or 32).")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Dataloader workers per GPU.")

    args = parser.parse_args()
    # Enforce 1 layer as per requirement
    if args.n_layer != 1:
        print("Warning: Overriding n_layer to 1 as per requirement.")
        args.n_layer = 1
    return args

class LitGPT(pl.LightningModule):
    """
    PyTorch Lightning Module for the GPT model.

    Encapsulates the model architecture, training step, validation step (optional),
    and optimizer configuration.
    """
    def __init__(self, config, tokenizer_vocab_size, learning_rate, warmup_steps, total_steps):
        super().__init__()
        # Save hyperparameters for checkpointing and logging
        #self.save_hyperparameters(ignore=['config']) # Avoid saving the large config object directly if not needed
        self.save_hyperparameters()

        self.config = config
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps # Needed for scheduler

        # Instantiate the GPT-2 model with language modeling head
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """ Forward pass through the model. """
        # If labels are provided, the model automatically calculates the loss.
        # GPT2LMHeadModel handles shifting labels internally for causal LM.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids # Use input_ids as labels for standard LM training
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """ Defines the computation performed at every training step. """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.loss # The model returns loss when labels are provided

        # Log training loss
        # 'prog_bar=True' shows it in the progress bar
        # 'logger=True' logs it using the configured logger (TensorBoard)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Defines the computation performed at every validation step. """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.loss

        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """ Configures the optimizer and learning rate scheduler. """
        # AdamW is a common choice for transformer models
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        # Linear warmup and decay scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        # Required format for PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Call scheduler step at every training step
                "frequency": 1
            }
        }


# Use Lightning's Dataset wrapper for compatibility if needed,
# but Hugging Face datasets often work directly. Let's use a simple wrapper.
class HFDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetches the sample and converts to tensors
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(item['attention_mask'], dtype=torch.long)
        }

class TinyStoriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the TinyStories dataset.

    Handles loading the preprocessed data, creating DataLoaders for
    training and validation splits.
    """
    def __init__(self, data_path: str, batch_size: int, block_size: int, num_workers: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size # Although chunking is done, good to have
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage: str = None):
        """ Load the dataset from disk. Called on each GPU separately. """
        # Load the processed dataset saved by preprocess_tinystories.py
        print(f"Loading dataset from: {self.data_path}")
        processed_dataset = load_from_disk(self.data_path)

        # Ensure splits exist
        if "train" not in processed_dataset or "validation" not in processed_dataset:
             raise ValueError(f"Dataset at {self.data_path} must contain 'train' and 'validation' splits.")

        # Assign to attributes used in dataloaders
        # Wrap in a simple torch Dataset for clarity
        self.train_dataset = HFDataset(processed_dataset["train"])
        self.val_dataset = HFDataset(processed_dataset["validation"])
        print(f"Dataset loaded. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")


    def train_dataloader(self):
        """ Returns the DataLoader for the training set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True, # Speeds up data transfer to GPU
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """ Returns the DataLoader for the validation set. """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

def main():
    """ Main function to set up and run the training process. """
    args = parse_args()
    pl.seed_everything(42) # for reproducibility

    # 1. Load Tokenizer (needed for vocab size)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # 2. Setup DataModule
    data_module = TinyStoriesDataModule(
        data_path=args.tokenized_data_path,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_workers=args.num_workers
    )
    # Important: Call setup manually here to calculate total_steps before trainer.fit
    data_module.setup()

    # Calculate total training steps for the scheduler
    # Needed because Lightning calculates this internally *during* fit, but we need it *before* for the scheduler setup
    effective_batch_size = args.batch_size * args.accumulate_grad_batches * args.gpus
    # Use len(data_module.train_dataloader()) if available, otherwise estimate
    # The dataloader length depends on the effective batch size after DDP,
    # so estimating based on dataset size is more robust here.
    steps_per_epoch = len(data_module.train_dataset) // effective_batch_size
    if len(data_module.train_dataset) % effective_batch_size != 0:
        steps_per_epoch += 1 # Account for the last potentially smaller batch
    total_steps = steps_per_epoch * args.num_epochs
    print(f"Effective Batch Size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")


    # 3. Setup Model (LightningModule)
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer, # Fixed to 1
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, # Set pad token id
        # Add other relevant GPT-2 config parameters if needed
    )

    lit_model = LitGPT(
        config=model_config,
        tokenizer_vocab_size=vocab_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps
    )

    # 4. Setup Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"{args.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=2,           # Save the best 2 models based on validation loss
        monitor="val_loss",     # Metric to monitor
        mode="min",             # Save the model with the minimum validation loss
        save_last=True          # Also save the latest checkpoint
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3, # Stop if validation loss doesn't improve for 3 epochs
        verbose=True,
        mode="min"
    )
    logger = TensorBoardLogger(save_dir=args.output_dir, name="logs")

    # 5. Setup Trainer
    # Use 'ddp' strategy for multi-GPU training
    # Ensure PyTorch distributed backend is initialized (usually handled by Lightning)
    # Use 'accelerator="gpu"' and 'devices=args.gpus' for newer PL versions
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_true" if args.gpus > 1 else None, # DDP strategy, find_unused helps with some models
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        # gradient_clip_val=1.0, # Optional: Gradient clipping
        log_every_n_steps=50   # How often to log metrics
    )

    # 6. Start Training
    print("Starting training...")
    trainer.fit(lit_model, datamodule=data_module)
    print("Training finished!")

    # Optional: Save the final Hugging Face model for easier inference later
    final_model_path = os.path.join(args.output_dir, "final_model")
    # Need to load the best checkpoint first to save the best model weights
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from checkpoint: {best_model_path}")
        best_lit_model = LitGPT.load_from_checkpoint(best_model_path)
        print(f"Saving final Hugging Face model to: {final_model_path}")
        best_lit_model.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path) # Save tokenizer with the model
    else:
        print("Could not find best model checkpoint path. Saving the last state.")
        lit_model.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
