# worker.Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install the specific TensorFlow TPU wheel.
RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl

# Download and install the TPU driver library. This is critical for the worker.
RUN apt-get update && apt-get install -y curl && \
    curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.9.0/libtpu.so -o /lib/libtpu.so && \
    ldconfig

COPY worker.py .

CMD ["python", "worker.py"]
