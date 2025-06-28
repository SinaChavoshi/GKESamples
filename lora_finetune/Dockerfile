# Use a PyTorch CPU base image, as TPUs do not use CUDA.
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

# Versions
ARG OPTIMUM_TPU='0.2.3'
ARG TRANSFORMERS='4.46.3'
ARG PEFT='0.13.2'
ARG TRL='0.12.1'
ARG DATASETS='3.1.0'
ARG ACCELERATE='1.1.0'
ARG EVALUATE='0.4.3'
ARG SAFETENSORS='0.4.5'

# Install dependencies
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir transformers[sklearn,sentencepiece]==${TRANSFORMERS}
RUN pip install --upgrade --no-cache-dir datasets==${DATASETS}
RUN pip install --upgrade --no-cache-dir accelerate==${ACCELERATE}
RUN pip install --upgrade --no-cache-dir evaluate==${EVALUATE}
RUN pip install --upgrade --no-cache-dir peft==${PEFT}
RUN pip install --upgrade --no-cache-dir trl==${TRL}
RUN pip install --upgrade --no-cache-dir safetensors==${SAFETENSORS}
RUN pip install --upgrade --no-cache-dir optimum-tpu==${OPTIMUM_TPU}
RUN pip install --upgrade --no-cache-dir gcsfs

# Set a working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .

# Command to run the training script
CMD ["python", "train.py"]
