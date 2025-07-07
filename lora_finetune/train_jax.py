import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
import flax
from flax.training import train_state
import optax
import transformers
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
import os
import argparse
import logging
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def create_learning_rate_fn(num_train_steps, learning_rate):
    """Creates a linear learning rate decay schedule."""
    warmup_steps = num_train_steps // 10
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_train_steps - warmup_steps,
        end_value=0.0,
    )

def main(args):
    # Authenticate with Hugging Face
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found")

    # Initialize JAX for multi-device environment
    jax.distributed.initialize()
    log.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    log.info(f"JAX local devices: {jax.local_devices()}")

    # --- 1. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = FlaxAutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=jnp.bfloat16,
    )

    # --- 2. Apply LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    # Note: PEFT for JAX/Flax is still evolving.
    # We get a PeftModel and then access its state for training.
    lora_model = PeftModelForCausalLM(model, lora_config)

    # --- 3. Load and Preprocess Dataset ---
    log.info("Loading pre-processed dataset from GCS...")
    # Use the same preprocessed data from your PyTorch script
    processed_dataset = load_from_disk(args.dataset_path)
    log.info("Dataset loaded successfully.")

    # --- 4. Setup Training State ---
    num_train_steps = len(processed_dataset) // args.batch_size * args.num_epochs
    learning_rate_fn = create_learning_rate_fn(num_train_steps, args.learning_rate)

    optimizer = optax.adamw(learning_rate=learning_rate_fn)

    # State holds the model parameters, optimizer, and other info
    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=lora_model.__call__,
        params=lora_model.params,
        tx=optimizer,
    )

    # --- 5. Define the Training Step ---
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, train=True)[0]
            # Simple cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"]).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # JIT compile the training step for immense speedup
    p_train_step = jax.jit(train_step)

    # --- 6. The Training Loop ---
    log.info("Starting JAX LoRA fine-tuning...")
    for epoch in range(args.num_epochs):
        log.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        for i in range(0, len(processed_dataset), args.batch_size):
            batch = processed_dataset[i:i+args.batch_size]
            
            # Manually create a batch suitable for JAX
            model_inputs = tokenizer(batch['text'], return_tensors="np", padding="max_length", truncation=True, max_length=1024)
            model_inputs["labels"] = model_inputs["input_ids"]

            state, loss = p_train_step(state, model_inputs)
            
            if i % (10 * args.batch_size) == 0: # Log every 10 batches
                log.info(f"Step {i // args.batch_size}, Loss: {loss}")

    log.info("Training complete.")

    # --- 7. Save the final adapter ---
    log.info(f"Saving final adapter to {args.output_dir}...")
    lora_model.save_pretrained(args.output_dir, params=state.params)
    log.info("LoRA adapter saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using a pre-converted Flax model to avoid conversion errors
    parser.add_argument("--model_id", type=str, default="benjamin/Llama-3.1-8B-Instruct-flax")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
