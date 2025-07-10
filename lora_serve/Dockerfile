FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

RUN pip install --upgrade pip

# The peft library is no longer needed for inference
RUN pip install --no-cache-dir \
    "transformers" \
    "accelerate" \
    "bitsandbytes"

WORKDIR /app

COPY chat.py .

CMD ["python", "chat.py"]
