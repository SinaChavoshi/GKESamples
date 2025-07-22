# Dockerfile
# This Dockerfile builds an image that contains both the coordinator and worker scripts.
FROM python:3.10-slim

WORKDIR /app

# Install git, which may be needed for some package dependencies.
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install the specific nightly build of TPU-enabled TensorFlow from the working example.
# This is a critical step.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install https://storage.googleapis.com/tensorflow-public-build-artifacts/prod/tensorflow/official/release/nightly/linux_x86_tpu/wheel_py310/749/20240915-062017/github/tensorflow/build_output/tf_nightly_tpu-2.18.0.dev20240915-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

# Copy both the coordinator and worker scripts into the container.
# The JobSet YAML will decide which script to run.
COPY main.py .
COPY start_worker.py .

# No CMD is specified, as it will be provided in the JobSet YAML.

