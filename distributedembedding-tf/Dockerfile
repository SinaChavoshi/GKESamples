# Use a specific, stable PyTorch CPU base image.
FROM pytorch/pytorch:2.2.1-cpu-py3

# Install dependencies in separate layers for better caching
RUN pip install --no-cache-dir "transformers==4.41.2"
RUN pip install --no-cache-dir "peft==0.11.1"
RUN pip install --no-cache-dir "trl==0.9.4"
RUN pip install --no-cache-dir "datasets==2.19.1"
RUN pip install --no-cache-dir "bitsandbytes==0.43.1"
RUN pip install --no-cache-dir "accelerate==0.30.1"
RUN pip install --no-cache-dir "gcsfs"
RUN pip install --no-cache-dir "pyarrow"

# Now, install PyTorch/XLA for TPU support. This is the last layer,
# as it's the least likely to change.
RUN pip install --no-cache-dir torch~=2.2.0 torch_xla~=2.2.0 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2-cp310-cp310-linux_x86_64.whl

# Set a working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .

# Use the full path to xla_dist as the entrypoint to ensure it's found.
# The command and args will be provided in the Kubernetes Job manifest.
ENTRYPOINT ["/usr/local/bin/xla_dist"]
