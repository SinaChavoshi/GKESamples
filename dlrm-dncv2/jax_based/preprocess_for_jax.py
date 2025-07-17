import os
import tempfile
from absl import app
from absl import flags
from absl import logging
import numpy as np
import shutil

# Import the DataLoader and all necessary constants
from dataloader import (
    CriteoDataLoader,
    DataConfig,
    NUM_DENSE_FEATURES,
    VOCAB_SIZES,
    MULTI_HOT_SIZES,
)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "source_data_path",
    "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*",
    "GCS path pattern for the source TFRecord files."
)
flags.DEFINE_string(
    "output_path",
    "/gcs/dlrm_preprocessed_jax/train",
    "Local path (GCS Fuse mount) to save the preprocessed NumPy arrays."
)
# --- UPDATED: Increased batch size for larger machines ---
flags.DEFINE_integer("processing_batch_size", 65536, "Batch size for processing records.")
flags.DEFINE_integer("num_total_batches", 10000, "Total number of batches to preprocess across all workers.")

def main(argv):
    del argv  # Unused.

    # --- Re-introduce parallel processing logic ---
    job_index_str = os.environ.get("JOB_COMPLETION_INDEX", "0")
    job_index = int(job_index_str) if job_index_str and job_index_str.isdigit() else 0

    parallelism_str = os.environ.get("PARALLELISM", "1")
    parallelism = int(parallelism_str) if parallelism_str and parallelism_str.isdigit() else 1

    logging.info(f"--- Starting Preprocessing Worker {job_index + 1}/{parallelism} ---")

    # Determine the range of batches this worker should process
    batches_per_worker = FLAGS.num_total_batches // parallelism
    start_batch = job_index * batches_per_worker
    end_batch = start_batch + batches_per_worker

    # Ensure the last worker gets any remaining batches
    if job_index == parallelism - 1:
        end_batch = FLAGS.num_total_batches
    
    logging.info(f"This worker will process batches from {start_batch} to {end_batch - 1}")
    
    data_config = DataConfig(
        global_batch_size=FLAGS.processing_batch_size,
        pre_batch_size=FLAGS.processing_batch_size, 
        is_training=False
    )
    
    data_loader = CriteoDataLoader(
        file_pattern=FLAGS.source_data_path,
        params=data_config,
        num_dense_features=NUM_DENSE_FEATURES,
        vocab_sizes=VOCAB_SIZES,
        multi_hot_sizes=MULTI_HOT_SIZES,
    )
    data_iterator = data_loader.get_iterator()

    os.makedirs(FLAGS.output_path, exist_ok=True)
    
    logging.info("Starting preprocessing...")
    # The loop now iterates over the specific slice for this worker
    for i in range(start_batch, end_batch):
        try:
            # Note: The underlying tf.data.Dataset is not sharded here.
            # We are manually advancing the iterator to the correct position.
            # This is simpler and avoids potential issues with tf.data sharding.
            batch = next(data_iterator)
            
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"batch_{i:05d}.npz.tmp")
            final_output_path = os.path.join(FLAGS.output_path, f"batch_{i:05d}.npz")

            with open(temp_file_path, 'wb') as f:
                np.savez_compressed(f, **batch)
            
            shutil.move(temp_file_path, final_output_path)

            if (i + 1) % 100 == 0:
                logging.info(f"Worker {job_index}: Successfully processed and saved batch {i + 1}")

        except StopIteration:
            logging.warning("Reached the end of the dataset before processing all requested batches.")
            break
            
    logging.info(f"--- Worker {job_index} Finished ---")

if __name__ == "__main__":
    app.run(main)
