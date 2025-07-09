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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def test_gcs_write(output_dir):
    """
    Tests authentication and write access to the specified GCS path.
    """
    logger.info("--- GCS WRITE-ACCESS AND AUTHENTICATION TEST ---")
    try:
        credentials, project = google.auth.default()
        logger.info(f"✅ Successfully authenticated with Google Cloud. Project ID: {project}")
        
        storage_client = storage.Client(credentials=credentials)
        bucket_name, blob_prefix = output_dir.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        test_blob_name = os.path.join(blob_prefix, "gcs_write_test.txt")
        blob = bucket.blob(test_blob_name)
        
        test_content = f"GCS write test successful at {time.time()}"
        blob.upload_from_string(test_content)
        
        logger.info(f"✅ Successfully wrote test file to gs://{bucket_name}/{test_blob_name}")
        logger.info("--- GCS TEST PASSED ---")
        return True

    except Exception as e:
        logger.error(f"❌ GCS write-access test FAILED.", exc_info=True)
        return False

def upload_directory_to_gcs(local_path, gcs_path):
    """
    Uploads the contents of a local directory to a GCS path.
    """
    logger.info(f"--- UPLOADING MODEL FROM LOCAL ('{local_path}') TO GCS ('{gcs_path}') ---")
    try:
        storage_client = storage.Client()
        bucket_name, gcs_prefix = gcs_path.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)

        for local_root, _, local_files in os.walk(local_path):
            for local_file in local_files:
                local_file_path = os.path.join(local_root, local_file)
                
                # Create a relative path to maintain the directory structure
                relative_path = os.path.relpath(local_file_path, local_path)
                gcs_blob_name = os.path.join(gcs_prefix, relative_path)
                
                blob = bucket.blob(gcs_blob_name)
                
                logger.info(f"Uploading '{local_file_path}' to 'gs://{bucket_name}/{gcs_blob_name}'")
                blob.upload_from_filename(local_file_path)

        logger.info(f"✅ Successfully uploaded all model files to GCS.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload directory to GCS: {e}", exc_info=True)
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on GKE with TPU using Optimum TPU.")
    # Keep the GCS path for the final destination
    parser.add_argument("--gcs_output_dir", type=str, required=True, help="The final GCS path to save the output adapter.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The GCS path to the *processed* dataset directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per TPU core.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length.")
    args = parser.parse_args()

    # Define a local temporary directory for model saving
    local_output_dir = "/tmp/lora_output"
    os.makedirs(local_output_dir, exist_ok=True)

    # Run GCS write test on the final destination before starting
    if not test_gcs_write(args.gcs_output_dir):
        time.sleep(60)
        sys.exit(1)

    logger.info("GCS access confirmed. Proceeding with training.")
    
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
    processed_dataset = load_from_disk(args.dataset_path)
    processed_dataset.set_format("torch")

    lora_config = LoraConfig(
        r=256, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"
    )

    # --- Configure SFT to save to the LOCAL directory ---
    sft_config = SFTConfig(
        output_dir=local_output_dir, # <-- Save locally
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
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training successfully completed.")

    logger.info(f"Saving final LoRA adapter to local path: {local_output_dir}")
    trainer.save_model(local_output_dir)
    logger.info(f"Adapter saved locally.")

    # --- Manually upload the saved model to GCS ---
    if not upload_directory_to_gcs(local_output_dir, args.gcs_output_dir):
        logger.error("Job finished with an error during GCS upload.")
        time.sleep(60)
        sys.exit(1)
        
    logger.info("Job finished successfully. Waiting for 15 seconds before exiting...")
    time.sleep(15)


if __name__ == "__main__":
    main()
