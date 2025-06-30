# Use the official Google-provided base image for PyTorch on TPU VMs.
# This image has torch and torch_xla pre-installed and configured correctly.
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Install your application dependencies in separate layers for better caching.
# Downgraded transformers to a version compatible with the Llama 3.1 config to fix the rope_scaling error.
RUN pip install --no-cache-dir "transformers==4.41.2"
RUN pip install --no-cache-dir "peft==0.11.1"
RUN pip install --no-cache-dir "trl==0.9.4"
RUN pip install --no-cache-dir "datasets==2.19.1"
RUN pip install --no-cache-dir "bitsandbytes==0.43.1"
RUN pip install --no-cache-dir "accelerate==0.30.1"
RUN pip install --no-cache-dir "gcsfs"
RUN pip install --no-cache-dir "pyarrow"

# Set a working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .

# No ENTRYPOINT or CMD is needed. We will specify the command
# directly in the Kubernetes Job manifest for clarity and flexibility.

