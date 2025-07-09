# Start from the same stable base image used for training
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Install the additional dependencies required by the chat script
# Note: Using the base image avoids torch/torch_xla installation issues

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    "peft" \
    "accelerate" \
    "bitsandbytes"

# Set the working directory inside the container
WORKDIR /app

# Copy the local chat script and the adapter directory into the container
COPY chat.py .
COPY my-lora-adapter/ ./my-lora-adapter/

# Define the command to run when the container starts
# This will start the interactive chat script by default
CMD ["python", "chat.py", "--lora_adapter_path", "./my-lora-adapter"]
