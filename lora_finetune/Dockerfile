FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Install the additional dependencies required by the training script
RUN  pip install --upgrade pip
RUN pip install --no-cache-dir \
    "optimum-tpu" -f https://storage.googleapis.com/libtpu-releases/index.html \
    "trl" \
    "peft" \
    "datasets" \
    "accelerate" \
    "evaluate"

# Set working directory
WORKDIR /app

# Copy your training script into the container
COPY train.py .
# Define the command to run your training script
CMD ["python", "train.py"]
