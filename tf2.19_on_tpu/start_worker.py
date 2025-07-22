# start_worker.py
# This script runs on each of the TPU pods.
# Its only job is to start a TensorFlow gRPC server and wait.
import os
import sys
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.training import server_lib

print("--- GRPC WORKER SCRIPT START ---")

try:
    # This is a standard way to configure a TF server.
    # It listens on port 8470 for instructions from the coordinator.
    server_def = tensorflow_server_pb2.ServerDef(protocol='grpc')
    job_def = server_def.cluster.job.add()
    job_def.name = 'tpu_worker'
    job_def.tasks[0] = 'localhost:8470'
    server_def.job_name = 'tpu_worker'
    server_def.task_index = 0

    config = config_pb2.ConfigProto()

    print("✅ Starting TensorFlow gRPC server...")
    server = server_lib.Server(server_def, config=config)
    print("✅ Server started. Waiting for connections...")

    # server.join() is a blocking call that keeps the server running indefinitely.
    server.join()

except Exception as e:
    print(f"🛑 GRPC WORKER FAILED: {e}")
    sys.exit(1)

