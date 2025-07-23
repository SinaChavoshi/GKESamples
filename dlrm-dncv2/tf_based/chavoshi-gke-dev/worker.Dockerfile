# worker.Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install the exact TPU release candidate found in the working image
RUN pip install --no-cache-dir tensorflow-tpu==2.19.0rc0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html
RUN pip install --no-cache-dir tf-keras tensorflow-datasets pyyaml gin-config keras-rs

# Copy and set up the worker script
COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh
ENTRYPOINT ["/usr/local/bin/tpu_tfjob_worker.sh"]
