# master.Dockerfile
FROM python:3.10-slim

# Set ENV vars discovered by the inspector
ENV GCS_CLIENT_CACHE_TYPE="None"
ENV GCS_READ_CACHE_MAX_SIZE_MB="0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB="0"
ENV TPU_STDERR_LOG_LEVEL="0"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install the exact TPU release candidate and other dependencies
RUN pip install --no-cache-dir tensorflow-tpu==2.19.0rc0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html
RUN pip install --no-cache-dir tf-keras tensorflow-datasets pyyaml gin-config keras-rs

# Clone the git repos discovered by the inspector
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/tensorflow/recommenders.git /recommenders && \
    git clone https://github.com/tensorflow/models.git /models
ENV PYTHONPATH="${PYTHONPATH}:/recommenders:/models"

# Add and set up the entrypoint script
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
