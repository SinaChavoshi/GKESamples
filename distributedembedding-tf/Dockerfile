# Use a specific, smaller base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install each dependency in its own layer.
# This creates more layers but allows Docker to cache each installation
# separately. The --no-cache-dir flag is used to keep image size down.

# Install standard Python packages
RUN pip install --no-cache-dir absl-py
RUN pip install --no-cache-dir tensorflow
RUN pip install --no-cache-dir tensorflow-datasets
RUN pip install --no-cache-dir "keras==3.10.0"
RUN pip install --no-cache-dir "keras-rs==0.2.1"

# Install JAX dependencies with the specific TPU release URL
RUN pip install --no-cache-dir "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir jax-tpu-embedding -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


# Copy your application code.
# This is the part that changes most often.
COPY train_jax.py .

# Set the command to run when the container starts
CMD ["python", "train_jax.py"]
