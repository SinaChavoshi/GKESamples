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
import os
import logging
from tqdm import tqdm

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(index, args):
    """Main training function."""
    
    # --- 1. Initialize PyTorch/XLA for TPU training ---
    device = xm.xla_device()
    
    # --- 2. Load Tokenizer and Model ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    logger.info(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading PyTorch model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16 # Use bfloat16 for TPU efficiency
    )

    # --- 3. Configure LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target modules for GPT-Neo
        task_type="CAUSAL_LM",
    )
    
    logger.info("Applying LoRA configuration to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.to(device)

    # --- 4. Load Pre-processed Dataset ---
    logger.info(f"Loading pre-processed dataset from GCS path: {args.dataset_path}")
    processed_dataset = load_from_disk(args.dataset_path)
    processed_dataset.set_format("torch")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        processed_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # --- 5. Set up Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # --- 6. Start the Training Loop ---
    logger.info("Starting training...")
    
    # Create the parallel loader
    para_loader = pl.ParallelLoader(train_dataloader, [device])
    
    for epoch in range(args.num_train_epochs):
        model.train()
        
        # Use tqdm only on the master process
        if xm.is_master_ordinal():
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for batch in para_loader.per_device_loader(device):
            # The labels are the input_ids themselves
            batch["labels"] = batch["input_ids"]
            
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            xm.optimizer_step(optimizer)
            
            if xm.is_master_ordinal():
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
        
        if xm.is_master_ordinal():
            pbar.close()

    # --- 7. Save the Final LoRA Adapter ---
    # Use xm.rendezvous to wait for all processes to finish before saving
    def _save_model_fn(model, output_dir):
        model.save_pretrained(output_dir)

    xm.rendezvous("save_model", _save_model_fn, args=(model, args.output_dir))
    
    if xm.is_master_ordinal():
        logger.info(f"LoRA adapter saved to GCS path: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="EleutherAI/gpt-neo-1.3B", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The GCS path to the *processed* dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="The GCS path to save the final LoRA adapter.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--per_device_batch_size", type=int, default=4, help="Batch size per TPU core.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    
    args = parser.parse_args()
    xmp.spawn(main, args=(args,))
