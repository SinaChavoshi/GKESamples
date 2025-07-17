import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from google.cloud import storage

from dataloader import CriteoDataLoader
from dataloader import DataConfig

FLAGS = flags.FLAGS

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    # This default will be inside the GCS Fuse mount in the pod
    "/gcs/dlrm_preprocessed_jax",
    "The GCS directory to save the preprocessed NumPy files.",
)
_NUM_BATCHES = flags.DEFINE_integer(
    "num_batches",
    10,
    "Number of batches to preprocess and save per worker.",
)

def main(argv):
  del argv

  # These will be set by the Kubernetes Job manifest
  # The Python script is now robust to an empty string for the index.
  parallelism = int(os.environ.get("PARALLELISM") or 1)
  job_index = int(os.environ.get("JOB_COMPLETION_INDEX") or 0)

  print(f"--- Starting Preprocessing Worker {job_index}/{parallelism} ---")

  data_config = DataConfig(
      global_batch_size=32768,
      pre_batch_size=4224,
      is_training=True,
      use_cached_data=False,
  )

  # Using the public GCS bucket for the source data
  file_pattern = "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*"

  data_loader = CriteoDataLoader(
      file_pattern=file_pattern,
      params=data_config,
      # These parameters match the benchmark
      num_dense_features=13,
      vocab_sizes=[
          40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
          3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
          40000000, 40000000, 590152, 12973, 108, 36
      ],
      multi_hot_sizes=[
          3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100,
          27, 10, 3, 1, 1
      ],
      # Pass sharding info to the data loader
      num_shards=parallelism,
      shard_index=job_index,
  )

  output_path_train = os.path.join(_OUTPUT_DIR.value, "train")
  output_path_eval = os.path.join(_OUTPUT_DIR.value, "eval")

  # Ensure the output directory exists
  os.makedirs(output_path_train, exist_ok=True)
  os.makedirs(output_path_eval, exist_ok=True)
  
  print(f"Worker {job_index}: Reading data and processing {_NUM_BATCHES.value} batches...")
  
  data_iterator = data_loader.get_iterator()

  for i in range(_NUM_BATCHES.value):
    try:
        batch = next(data_iterator)
        
        # Each worker saves its files with a unique name
        output_file = os.path.join(
            output_path_train, f"batch_{job_index}_{i}.npz"
        )
        
        # Save the dictionary of numpy arrays
        # Use np.savez_compressed for efficiency
        np.savez_compressed(output_file, **batch)

        if i % 10 == 0:
            print(f"Worker {job_index}: Saved {i+1}/{_NUM_BATCHES.value} batches...")

    except StopIteration:
      print(f"Worker {job_index}: Reached end of dataset.")
      break

  print(f"--- Worker {job_index} finished successfully ---")

if __name__ == "__main__":
  app.run(main)
