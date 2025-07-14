# Use the official public Python base image.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the nightly Keras and the compatible nightly TPU-enabled TensorFlow.
# This ensures the TensorFlow version matches the TPU library version.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    gin-config \
    tensorflow-datasets
RUN pip install tf-keras-nightly --no-deps
RUN pip install --no-cache-dir tf-nightly-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html

RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Clone the required GitHub repositories
RUN git clone https://github.com/tensorflow/recommenders.git && \
    git clone https://github.com/tensorflow/models.git

# Set the PYTHONPATH so the training script can find the libraries
ENV PYTHONPATH="/app/recommenders/:/app/models/"

# Copy the training script into the container
COPY run_training.sh .
RUN chmod +x run_training.sh

# Set the entrypoint for the container
CMD ["./run_training.sh"]
