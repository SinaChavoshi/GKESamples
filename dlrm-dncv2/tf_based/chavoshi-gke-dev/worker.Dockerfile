
# worker image
FROM python:3.12
RUN pip install \
   --no-cache-dir \
   --upgrade \
   pip
RUN pip install --no-cache-dir tensorflow-tpu==2.18.0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
# RUN pip install --no-cache-dir tensorflow-tpu==2.19.0rc0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
# RUN pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.13.0/tensorflow-2.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.7.0/libtpu.so -o /lib/libtpu.so
# RUN pip uninstall -y libtpu
# RUN pip install libtpu==0.0.10.1
# ADD libtpu/20250108/tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#RUN pip install --no-cache-dir tf_nightly_tpu-2.19.0.dev20250108-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force
#COPY libtpu/20250108/libtpu.so /usr/local/lib/python3.10/site-packages/libtpu/libtpu.so
COPY tpu_tfjob_worker.sh /usr/local/bin/tpu_tfjob_worker.sh
RUN chmod +x /usr/local/bin/tpu_tfjob_worker.sh
ENTRYPOINT ["/usr/local/bin/tpu_tfjob_worker.sh"]
