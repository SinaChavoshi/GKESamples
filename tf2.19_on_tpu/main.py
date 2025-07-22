# main.py
# This script runs on the single CPU coordinator pod.
# It connects to the gRPC workers on the TPU pods to run the computation.
import os
import sys
import json
import ctypes

# This MUST be the first thing we do.
# It prevents TensorFlow from automatically loading libtpu.so and trying to
# access the metadata server before we have a chance to configure it.
os.environ['TPU_LOAD_LIBRARY'] = '0'

print("\n--- TPU COORDINATOR SCRIPT START ---")

# --- Step 1: Set ALL environment variables for libtpu.so initialization ---
# These variables provide the physical topology information directly,
# bypassing the need for the GCP metadata server.
print("\n--- Setting Environment Variables for libtpu.so ---")
jobset_name = os.environ.get('JOBSET_NAME')
grpc_worker_name = os.environ.get('GRPC_WORKER_NAME')
replicas_str = os.environ.get('NUM_REPLICAS')

if not all([jobset_name, grpc_worker_name, replicas_str]):
    print(f"JOBSET_NAME: {jobset_name}")
    print(f"GRPC_WORKER_NAME: {grpc_worker_name}")
    print(f"NUM_REPLICAS: {replicas_str}")
    print("\n🛑 ERROR: Missing one or more required environment variables for the coordinator.")
    sys.exit(1)

replicas = int(replicas_str)

# Construct the hostnames (without grpc:// or port)
worker_hostnames_list = []
for i in range(replicas):
    hostname = f"{jobset_name}-{grpc_worker_name}-{i}-0.{jobset_name}"
    worker_hostnames_list.append(hostname)
worker_hostnames_string = ",".join(worker_hostnames_list)

os.environ['TPU_WORKER_HOSTNAMES'] = worker_hostnames_string
os.environ['TPU_ACCELERATOR_TYPE'] = 'tpu-v4-podslice'
os.environ['TPU_WORKER_ID'] = '0' # The coordinator acts as the first logical worker for initialization.
os.environ['HOST_BOUNDS'] = '2,2,2'
os.environ['CHIPS_PER_HOST_BOUNDS'] = '2,2,1'

print(f"TPU_WORKER_HOSTNAMES: {os.environ['TPU_WORKER_HOSTNAMES']}")
print(f"TPU_ACCELERATOR_TYPE: {os.environ['TPU_ACCELERATOR_TYPE']}")
print(f"TPU_WORKER_ID: {os.environ['TPU_WORKER_ID']}")
print(f"HOST_BOUNDS: {os.environ['HOST_BOUNDS']}")
print(f"CHIPS_PER_HOST_BOUNDS: {os.environ['CHIPS_PER_HOST_BOUNDS']}")


# --- Step 2: Manually load the TPU library ---
# This is the final critical step. We load the library ourselves now that the
# environment is correctly configured.
print("\n--- Attempting to explicitly load libtpu.so ---")
libtpu_path = '/usr/local/lib/python3.10/site-packages/libtpu/libtpu.so'
try:
    ctypes.CDLL(libtpu_path)
    print("✅ Successfully loaded libtpu.so.")
except Exception as e:
    print(f"🛑 CRITICAL ERROR: Could not load libtpu.so from path: {libtpu_path}")
    print(f"    Error details: {e}")
    sys.exit(1)


# --- Step 3: Now import TensorFlow and build the gRPC endpoint list ---
import tensorflow as tf

worker_endpoints = []
for hostname in worker_hostnames_list:
    endpoint = f"grpc://{hostname}:8470"
    worker_endpoints.append(endpoint)
endpoint_string = ",".join(worker_endpoints)
print(f"\nConstructed gRPC Endpoints for Resolver: {endpoint_string}")


# --- Step 4: Attempt TPU Initialization ---
print("\n--- TPU Initialization Attempt ---")
try:
    tf.config.set_visible_devices([], 'GPU')
    print("✅ Disabled local GPU devices.")

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=endpoint_string)

    print("✅ Connecting to cluster...")
    tf.config.experimental_connect_to_cluster(resolver)

    print("✅ Initializing TPU system...")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("✅ TPU system initialization SUCCEEDED!")

    all_devices = tf.config.list_logical_devices('TPU')
    print(f"\nAll available devices ({len(all_devices)}): {all_devices}")

    # --- Step 5: Run the actual debug test ---
    print("\n--- Manual Device Placement Test ---")
    strategy = tf.distribute.TPUStrategy(resolver)
    with strategy.scope():
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)

    print("\n--- Output ---")
    print(f"Device of resulting tensor 'c': {c.device}")
    print(f"Value of 'c':\n {c.numpy()}")
    print("\n✅ Script completed successfully. Debug flags should be visible in the logs.")
    sys.exit(0)

except Exception as e:
    print(f"\n🛑 An error occurred during TPU initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- TPU COORDINATOR SCRIPT END ---")

