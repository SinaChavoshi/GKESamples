import functools
import time
from typing import Any
import os

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import gcsfs

# Import the new dataloader and model
from dlrm_model import DLRMDCNV2
import optax

# Initialize JAX distributed environment
jax.distributed.initialize()
# Start JAX profiler server for performance debugging
jax.profiler.start_server(9999)

FLAGS = flags.FLAGS

# --- Benchmark and Model Constants ---
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128

# --- Flags for Configuration ---
flags.DEFINE_string(
    "preprocessed_data_path", 
    "gs://your-bucket-name/dlrm_preprocessed_jax/train/*.npz", # <--- UPDATE THIS
    "GCS path pattern for the preprocessed NumPy batch files."
)
flags.DEFINE_integer("global_batch_size", 32768, "Global batch size for training.")
flags.DEFINE_integer("num_steps", 10000, "Number of training steps.")
flags.DEFINE_float("learning_rate", 0.00025, "Learning rate for Adagrad optimizer.")
flags.DEFINE_integer("log_frequency", 100, "Frequency to log training metrics.")


def load_preprocessed_data(path_pattern: str):
    """Loads preprocessed .npz files from GCS into a list of batches."""
    logging.info(f"Loading preprocessed data from: {path_pattern}")
    gcs = gcsfs.GCSFileSystem()
    file_paths = gcs.glob(path_pattern)
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {path_pattern}")

    all_batches = []
    for path in file_paths:
        with gcs.open(path, 'rb') as f:
            all_batches.append(dict(np.load(f)))
    logging.info(f"Loaded {len(all_batches)} preprocessed batches into memory.")
    return all_batches


def main(argv: Any):
    del argv  # Unused.

    # --- JAX Distributed Setup ---
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("data",))
    data_sharding = NamedSharding(mesh, P("data", None))
    model_sharding = NamedSharding(mesh, P()) 

    # --- Data Loading (from preprocessed files) ---
    train_batches = load_preprocessed_data(FLAGS.preprocessed_data_path)

    # --- Model and Optimizer Setup ---
    logging.info("Initializing DLRM-DCNv2 model and optimizer...")
    model = DLRMDCNV2(
        vocab_sizes=VOCAB_SIZES,
        embedding_dim=EMBEDDING_DIM,
        num_dense_features=NUM_DENSE_FEATURES,
    )

    per_device_batch_size = FLAGS.global_batch_size // jax.device_count()
    dummy_dense = jnp.zeros((per_device_batch_size, NUM_DENSE_FEATURES))
    dummy_sparse = {
        str(i): jnp.zeros((per_device_batch_size, 1), dtype=jnp.int32) 
        for i in range(len(VOCAB_SIZES))
    }
    
    params = model.init({"params": jax.random.PRNGKey(0)}, dummy_dense, dummy_sparse)["params"]
    replicated_params = jax.device_put(params, model_sharding)

    tx = optax.adagrad(learning_rate=FLAGS.learning_rate)
    opt_state = tx.init(replicated_params)
    replicated_opt_state = jax.device_put(opt_state, model_sharding)

    # --- JIT-Compiled Training Step ---
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            logits = model.apply({"params": p}, batch["dense_features"], batch["sparse_features"])
            loss = optax.sigmoid_binary_cross_entropy(logits, batch["clicked"]).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        
        grads = jax.lax.pmean(grads, axis_name="data")
        
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss

    # --- Training Loop ---
    logging.info("Starting training loop...")
    start_time = time.time()
    for step in range(FLAGS.num_steps):
        # Cycle through the in-memory list of batches
        batch = train_batches[step % len(train_batches)]
        
        # Convert numpy arrays to sharded JAX arrays for the current step
        sharded_batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), batch)

        replicated_params, replicated_opt_state, loss = train_step(
            replicated_params, replicated_opt_state, sharded_batch
        )

        if (step + 1) % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - start_time
            num_samples = FLAGS.log_frequency * FLAGS.global_batch_size
            throughput = num_samples / duration
            logging.info(
                f"Step: {step + 1}/{FLAGS.num_steps}, Loss: {loss:.4f}, "
                f"Throughput: {throughput:.2f} examples/sec"
            )
            start_time = current_time
    
    logging.info("Training complete.")

if __name__ == "__main__":
    app.run(main)
