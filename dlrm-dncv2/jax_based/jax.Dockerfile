# Use the official Python 3.12 image, as required.
FROM python:3.10-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install git, which is required for cloning the recml repository
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel and a modern version of setuptools first.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# This ensures that pip resolves a set of packages that are compatible with each other and the TPU runtime.
RUN pip install --no-cache-dir \
    'setuptools<60' \
    promise==2.3 \
    absl-py \
    etils \
    fiddle \
    flax \
    optax \
    jaxtyping \
    tensorflow \
    scikit-learn \
    clu \
    tensorflow-datasets

# Install the correct JAX version for TPUs
RUN pip install -U https://storage.googleapis.com/jax-tpu-embedding-whls/20250604/jax_tpu_embedding-0.1.0.dev20250604-cp310-cp310-manylinux_2_35_x86_64.whl --force
RUN pip install -U "jax[tpu]>0.4.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clone the recml repo to ensure the latest version is used.
# TODO: Please replace with the correct internal git URL for your project if needed.
RUN git clone https://github.com/AI-Hypercomputer/RecML.git /app/recml

# Add the cloned repository to the PYTHONPATH to make it importable.
ENV PYTHONPATH="/app/recml:${PYTHONPATH}"


# COPY dlrm_main.py recml/inference/models/jax/DLRM_DCNv2/

ENV PYTHONPATH="/app/recml:${PYTHONPATH}"
WORKDIR /app/recml/RecML
