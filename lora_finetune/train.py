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
import sys
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from optimum.tpu import fsdp_v2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on GKE with TPU using GCS FUSE.")
    # Arguments now expect local filesystem paths provided by the FUSE mount
    parser.add_argument("--output_dir", type=str, required=True, help="The local path to save the output model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The local path to the processed dataset directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per TPU core.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length.")
    args = parser.parse_args()

    # --- Standard Training Setup ---
    fsdp_v2.use_fsdp_v2()
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=hf_token, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load dataset from the local FUSE path
    logger.info(f"Loading dataset from local path: {args.dataset_path}")
    processed_dataset = load_from_disk(args.dataset_path)
    processed_dataset.set_format("torch")

    lora_config = LoraConfig(
        r=256, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"
    )

    # --- Configure SFT to save directly to the final mounted GCS path ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adafactor",
        dataloader_drop_last=True,
        max_seq_length=args.max_seq_length,
        packing=True,
        **fsdp_v2.get_fsdp_training_args(model),
    )

    trainer = SFTTrainer(
        model=model, train_dataset=processed_dataset,
        peft_config=lora_config, args=sft_config
    )

    model.config.use_cache = False

    logger.info(f"Starting training... Output will be saved to {args.output_dir}")
    trainer.train()
    logger.info("Training successfully completed.")

    logger.info("Saving FSDP-sharded LoRa adapter...")

    # On the main process, gather the sharded model state dictionary
    if trainer.is_world_process_zero():
        # Use the peft utility to get the correct state dict for the adapter
        state_dict = get_peft_model_state_dict(trainer.model)
        # Save the adapter using the gathered state dict
        trainer.model.save_pretrained(args.output_dir, state_dict=state_dict)

        # The tokenizer is not sharded, so it can be saved directly.
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"LoRa adapter and tokenizer saved successfully to {args.output_dir}")

    logger.info("Job finished successfully.")


if __name__ == "__main__":
    main()
