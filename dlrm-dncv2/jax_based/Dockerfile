# Use an official Google Cloud JAX base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install your specific dependencies
# The base image already has JAX, numpy, tensorflow, etc.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir absl-py
RUN pip install --no-cache-dir etils
RUN pip install --no-cache-dir fiddle
RUN pip install --no-cache-dir flax
RUN pip install --no-cache-dir optax
RUN pip install --no-cache-dir recml
RUN pip install --no-cache-dir jax[tpu]
RUN pip install --no-cache-dir jaxtyping



# Copy your application code into the container
COPY dlrm_experiment.py .
COPY main.py .

# Define the command to run when the container starts
CMD ["python", "main.py"]
