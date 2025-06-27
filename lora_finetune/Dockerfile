# Use a PyTorch base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir    "transformers"
RUN pip install --no-cache-dir    "peft"
RUN pip install --no-cache-dir    "trl"
RUN pip install --no-cache-dir    "datasets"
RUN pip install --no-cache-dir    "bitsandbytes"
RUN pip install --no-cache-dir    "accelerate"
RUN pip install --no-cache-dir    torch_xla[tpu]~=2.2.0 --extra-index-url https://storage.googleapis.com/pytorch-xla-releases/wheels
RUN pip install torch~=2.2.0 torch_xla~=2.2.0 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2-cp310-cp310-linux_x86_64.whl

# Set a working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .

# Command to run the training script
CMD ["python", "train.py"]
