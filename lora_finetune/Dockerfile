

# Use an official Google-provided JAX base image for TPUs
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/jax-tpu:2.0.0

# Install dependencies for JAX-based training
# Note: PEFT supports Flax for LoRA
RUN pip install --no-cache-dir "transformers>=4.43.1"
RUN pip install --no-cache-dir "peft>=0.11.1"
RUN pip install --no-cache-dir "datasets>=2.19.1"
RUN pip install --no-cache-dir "optax"
RUN pip install --no-cache-dir "gcsfs"
RUN pip install --no-cache-dir "pyarrow"

# Set a working directory
WORKDIR /app

# Copy the new JAX training script into the container
COPY train_jax.py .
