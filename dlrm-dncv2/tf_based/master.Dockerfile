# master image
FROM python:3.10

ENV GCS_CLIENT_CACHE_TYPE "None"
ENV GCS_READ_CACHE_MAX_SIZE_MB "0"
ENV GCS_READ_CACHE_BLOCK_SIZE_MB "0"
ENV TPU_STDERR_LOG_LEVEL "0"
ENV TF_USE_LEGACY_KERAS "1"

# Install TPU Tensorflow package
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
RUN pip install --no-cache-dir tf-keras==2.18.0 tensorflow-datasets pyyaml gin-config
RUN pip uninstall -y tf-nightly
RUN pip install --no-cache-dir tensorflow-tpu==2.18.0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
RUN pip install --force-reinstall "typing_extensions>=3.6.6,<4.6.0"
#RUN pip install --no-cache-dir tensorflow-tpu==2.19.0rc0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
#RUN pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.13.0/tensorflow-2.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#ADD 20250108/tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#RUN pip install --no-cache-dir tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
# RUN pip install --no-cache-dir tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

# Clone TFRS to a dir
RUN git clone https://github.com/tensorflow/recommenders.git /recommenders
ENV PYTHONPATH "${PYTHONPATH}:/recommenders"
RUN git clone https://github.com/ACW101/models.git /models
ENV PYTHONPATH "${PYTHONPATH}:/models"

# Add entrypoint.sh
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
