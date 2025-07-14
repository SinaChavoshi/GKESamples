# Use a standard Python image
FROM python:3.11-slim

WORKDIR /app

# Install git and other essentials
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install JAX, Keras 3, and TPU support libraries
# Note: keras-core is the package name for Keras 3
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir \
    keras-core \
    tensorflow \
    tensorflow-recommenders

# Copy the JAX training script into the container
COPY train_jax.py .

# Set the entrypoint for the container
# We will pass arguments via the jobset.yaml
CMD ["python3", "train_jax.py"]
