# Dockerfile
# Use a base image that is compatible with JAX and TPUs
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies one by one
RUN pip install --no-cache-dir absl-py
RUN pip install --no-cache-dir etils
RUN pip install --no-cache-dir fiddle
RUN pip install --no-cache-dir flax
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir optax
RUN pip install --no-cache-dir tensorflow

# Install JAX for Google Cloud TPU using the specific release URL
RUN pip install --no-cache-dir --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]

# Copy your application code into the container
COPY dlrm_experiment.py .
COPY main.py .

# Define the command to run when the container starts
CMD ["python", "main.py"]
