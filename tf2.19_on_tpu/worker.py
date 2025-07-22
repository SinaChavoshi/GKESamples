import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.training import server_lib

print("Starting TensorFlow gRPC server...")

# This server definition is minimal, as GKE's TPU environment handles
# the actual cluster configuration.
server_def = tensorflow_server_pb2.ServerDef(protocol='grpc')
job_def = server_def.cluster.job.add()
job_def.name = 'tpu_worker'
job_def.tasks[0] = 'localhost:8470'
server_def.job_name = 'tpu_worker'
server_def.task_index = 0
config = config_pb2.ConfigProto()

# Create and start the gRPC Server.
# The .join() method is blocking and will run forever.
server = server_lib.Server(server_def, config=config)
server.join()
