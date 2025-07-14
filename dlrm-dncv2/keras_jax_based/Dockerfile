# Use a standard Python image
FROM python:3.11-slim

WORKDIR /app

# Install git and other essentials
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install JAX, Keras, and TensorFlow (for the data pipeline)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir \
    keras-core \
    tensorflow \
    tensorflow-recommenders

COPY train_jax.py .

# Set the entrypoint for the JAX training
CMD ["python3", "train_jax.py"]
