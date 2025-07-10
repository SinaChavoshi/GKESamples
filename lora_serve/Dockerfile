# lora_serve/Dockerfile
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm

RUN pip install --upgrade pip

# Add the "peft" library back for inference
RUN pip install --no-cache-dir \
    "transformers" \
    "accelerate" \
    "bitsandbytes" \
    "peft"

WORKDIR /app

COPY chat.py .
CMD ["python", "chat.py"]
