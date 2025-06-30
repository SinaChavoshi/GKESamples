FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11

RUN pip install --no-cache-dir "jax[k8s]"

# Install dependencies for JAX-based training
RUN pip install --upgrade pip
RUN pip install --no-cache-dir "flax"
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
