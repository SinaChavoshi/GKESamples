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
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
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
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length.")
    args = parser.parse_args()

    # --- 1. Enable FSDPv2 for model sharding ---
    logger.info("Enabling FSDPv2 for model sharding.")
    fsdp_v2.use_fsdp_v2()

    # --- 2. Load Tokenizer and Model ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    logger.info(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    logger.info(f"Loading PyTorch model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 3. Load Preprocessed Dataset ---
    logger.info(f"Loading pre-processed dataset from GCS path: {args.dataset_path}")
    train_dataset = load_from_disk(args.dataset_path)
    train_dataset.set_format("torch")
    logger.info(f"Loaded {len(train_dataset)} training examples.")


    # --- 4. Configure LoRA and FSDP ---
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
    logger.info(f"FSDP Config: {fsdp_training_args}")


    # --- 5. Set up and run the SFTTrainer ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        packing=True,
        optim="adafactor",
        dataloader_drop_last=True,
        logging_steps=1,
        **fsdp_training_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    
    model.config.use_cache = False

    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training successfully completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Keep the pod alive for a bit to allow for log inspection
        time.sleep(60)
        raise e

    logger.info("Saving final LoRA adapter.")
    try:
        trainer.save_model(args.output_dir)
        logger.info(f"LoRA adapter saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}", exc_info=True)
        # Keep the pod alive for a bit to allow for log inspection
        time.sleep(60)
        raise e
    
    # Keep the pod alive for a short time to ensure GCS upload completes
    logger.info("Waiting for 15 seconds before exiting...")
    time.sleep(15)


if __name__ == "__main__":
    main()
