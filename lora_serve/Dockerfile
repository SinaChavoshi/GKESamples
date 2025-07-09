# Start from the same stable base image used for training
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Install the additional dependencies required by the chat script
RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    "peft" \
    "accelerate" \
    "bitsandbytes" \
    "gcsfs"

# Set the working directory inside the container
WORKDIR /app

# Copy the local chat script into the container
COPY chat.py .

# Define the command to run when the container starts
# The GCS path will be passed in via the Kubernetes deployment manifest
CMD ["python", "chat.py"]
