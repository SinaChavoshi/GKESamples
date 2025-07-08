# Use a PyTorch base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install Python dependencies for PyTorch/XLA TPU training
RUN pip install --no-cache-dir \
    "transformers>=4.38.0" \
    "datasets" \
    "peft" \
    "accelerate" \
    "tqdm" \
    "gcsfs" \
    "google-cloud-storage" \
    # Install the latest compatible torch_xla for TPU
    "torch_xla[tpu]"

# Set working directory
WORKDIR /app

# Copy your training script
COPY train.py .

# The command to run your training script
CMD ["python", "train.py"]
