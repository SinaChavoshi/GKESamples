# Use the official Google-provided base image for PyTorch on TPU VMs.
# This image has torch and torch_xla pre-installed and configured correctly.
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Install your application dependencies
RUN pip install --update "pip"
RUN pip install --no-cache-dir "transformers>=4.43.1"
RUN pip install --no-cache-dir "peft==0.11.1"
RUN pip install --no-cache-dir "trl==0.9.4"
RUN pip install --no-cache-dir "datasets==2.19.1"
RUN pip install --no-cache-dir "accelerate>=0.31.0"
RUN pip install --no-cache-dir "gcsfs"
RUN pip install --no-cache-dir "pyarrow"

# Set a working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .
