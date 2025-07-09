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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from optimum.tpu import fsdp_v2
import google.auth
from google.cloud import storage

# Configure logging to be very clear
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def test_gcs_write(output_dir):
    """
    Tests authentication and write access to the specified GCS path.
    Exits the script with an error if it fails.
    """
    logger.info("--- GCS WRITE-ACCESS AND AUTHENTICATION TEST ---")
    try:
        # 1. Test Authentication
        credentials, project = google.auth.default()
        logger.info(f"✅ Successfully authenticated with Google Cloud. Project ID: {project}")

        # 2. Test GCS Write
        logger.info(f"Attempting to write a test file to: {output_dir}")
        storage_client = storage.Client(credentials=credentials)
        
        # Parse bucket name and prefix from the GCS path
        if not output_dir.startswith("gs://"):
            raise ValueError("Output directory must be a GCS path (gs://...).")
        
        bucket_name, blob_prefix = output_dir.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        
        # Define the test file blob
        test_blob_name = os.path.join(blob_prefix, "gcs_write_test.txt")
        blob = bucket.blob(test_blob_name)
        
        # Write content to the test file
        test_content = f"GCS write test successful at {time.time()}"
        blob.upload_from_string(test_content)
        
        logger.info(f"✅ Successfully wrote test file to gs://{bucket_name}/{test_blob_name}")
        logger.info("--- GCS TEST PASSED ---")
        return True

    except Exception as e:
        logger.error(f"❌ GCS write-access test FAILED. The job will now terminate.", exc_info=True)
        return False


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

    # --- Run GCS write test before doing anything else ---
    if not test_gcs_write(args.output_dir):
        # Keep the pod alive for 60 seconds to allow log inspection
        time.sleep(60)
        sys.exit(1) # Exit with an error code

    # --- If the test passes, proceed with training ---
    logger.info("GCS access confirmed. Proceeding with model training.")
    
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
    processed_dataset = load_from_disk(args.dataset_path)
    processed_dataset.set_format("torch")

    # --- 4. Configure LoRA and FSDP ---
    lora_config = LoraConfig(
        r=256,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
    logger.info(f"FSDP Config: {fsdp_training_args}")

    # --- 5. Set up SFTConfig ---
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
        **fsdp_training_args,
    )

    # --- 6. Set up and run the SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        peft_config=lora_config,
        args=sft_config,
    )
    
    model.config.use_cache = False

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training successfully completed.")

    logger.info("Saving final LoRA adapter.")
    trainer.save_model(args.output_dir)
    logger.info(f"Final LoRA adapter saved to {args.output_dir}")
    
    logger.info("Job finished. Waiting for 15 seconds before exiting...")
    time.sleep(15)


if __name__ == "__main__":
    main()
