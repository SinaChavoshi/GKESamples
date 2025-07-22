# controller.Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.19.0/experimental/tensorflow_tpu-2.19.0-cp310-cp310-linux_x86_64.whl

RUN pip install --no-cache-dir tensorflow-datasets==4.9.4

COPY main.py .

CMD ["python", "main.py"]
