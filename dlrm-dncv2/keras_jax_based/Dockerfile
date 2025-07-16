FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir \
    keras \
    tensorflow \
    keras-rs

COPY train_jax.py .

CMD ["python3", "train_jax.py"]
