FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE "None"
ENV GCS_READ_CACHE_MAX_SIZE_MB "0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB "0"
ENV TF_USE_LEGACY_KERAS "1"

# This was added so that we can get libtpu logs on the worker. This is useful to debug errors during training.
ENV TPU_STDERR_LOG_LEVEL "0"

# TODO: Remove these env variables if they are not required after upgrading to TF2.13
ENV ENABLE_ICI_RESILIENCY "false"
ENV ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY "false"

RUN apt-get update && \
    apt-get install libtcmalloc-minimal4 && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

COPY dockerfiles/tpu-training/tpu-runtime.service /etc/systemd/system/tpu-runtime.service
COPY dockerfiles/tpu-training/tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh

# Install TPU Tensorflow package
RUN pip install \
    --no-cache-dir \
    --upgrade \
    pip
RUN pip install --no-cache-dir tf-keras==2.19.0 tensorflow-serving-api==2.19.0
RUN pip uninstall -y tensorflow
RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir libtpu==0.0.12

# Clone TFRS to a dir
RUN git clone https://github.com/tensorflow/recommenders.git /recommenders
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"

# Add entrypoint.sh
ADD dockerfiles/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
