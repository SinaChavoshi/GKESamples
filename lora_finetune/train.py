#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import logging
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
# We will use SFTConfig instead of TrainingArguments
from trl import SFTConfig, SFTTrainer
from optimum.tpu import fsdp_v2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on GKE with TPU using Optimum TPU.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The GCS path to the *processed* dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="The GCS path to save the output adapter.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per TPU core.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length.")
    args = parser.parse_args()

    # --- 1. Enable FSDPv2 for model sharding ---
    # This must be called at the beginning of the script.
    logger.info("Enabling FSDPv2 for model sharding.")
    fsdp_v2.use_fsdp_v2()

    # --- 2. Load Tokenizer and Model ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    logger.info(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    # Add a pad token if the model doesn't have one. Llama models often don't.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    logger.info(f"Loading PyTorch model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    # The model's pad_token_id must match the tokenizer's
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 3. Load Preprocessed Dataset ---
    logger.info(f"Loading pre-processed dataset from GCS path: {args.dataset_path}")
    processed_dataset = load_from_disk(args.dataset_path)
    processed_dataset.set_format("torch")

    # --- 4. Configure LoRA and FSDP ---
    # LoRA config for PEFT
    lora_config = LoraConfig(
        r=256,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear", # Automatically targets all linear layers
        task_type="CAUSAL_LM",
    )

    # Get FSDP arguments from Optimum TPU
    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
    logger.info(f"FSDP Config: {fsdp_training_args}")

    # --- 5. Set up SFTConfig ---
    # SFTConfig now holds all the training parameters
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adafactor", # Adafactor is often recommended for TPUs
        dataloader_drop_last=True,  # Required for FSDP
        max_seq_length=args.max_seq_length,
        packing=True, # Packs multiple short sequences into one for efficiency
        # Unpack FSDP config into the SFTConfig
        **fsdp_training_args,
    )


    # --- 6. Set up and run the SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        peft_config=lora_config,
        args=sft_config, # Pass the SFTConfig object here
    )
    
    # Disable cache for gradient checkpointing
    model.config.use_cache = False

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete. Saving final LoRA adapter.")
    trainer.save_model(args.output_dir)
    logger.info(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
