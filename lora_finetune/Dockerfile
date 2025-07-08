FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11


# Install dependencies for JAX-based training
RUN pip install --upgrade pip
RUN pip install --no-cache-dir "flax"
RUN pip install --no-cache-dir "transformers[flax]==4.41.2"
RUN pip install --no-cache-dir "peft==0.10.0"
RUN pip install --no-cache-dir "datasets>=2.19.1"
RUN pip install --no-cache-dir "optax"
RUN pip install --no-cache-dir "gcsfs"
RUN pip install --no-cache-dir "pyarrow"
RUN pip install --no-cache-dir "jax[tpu]==0.4.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Set a working directory
WORKDIR /app

# Copy the JAX training script into the container
COPY train_jax.py .
