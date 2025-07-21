FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir \
    tensorflow-cpu && \
    pip install --no-cache-dir keras-rs -f git+https://github.com/keras-team/keras-rs.git && \
    pip install --no-cache-dir keras -f git+https://github.com/keras-team/keras.git && \ 

COPY train_jax.py .

CMD ["python3", "train_jax.py"]

