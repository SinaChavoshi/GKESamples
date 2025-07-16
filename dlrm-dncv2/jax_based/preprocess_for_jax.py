import os
from absl import app
from absl import flags
from absl import logging
import numpy as np

# Import the same DataLoader used in your training script
from dataloader import CriteoDataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "source_data_path",
    "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*",
    "GCS path pattern for the source TFRecord files."
)
flags.DEFINE_string(
    "output_path",
    # Updated to use the GCS Fuse mount path and your bucket
    "/gcs/dlrm_preprocessed_jax/train",
    "Local path (GCS Fuse mount) to save the preprocessed NumPy arrays."
)
flags.DEFINE_integer("global_batch_size", 32768, "Global batch size for processing.")
flags.DEFINE_integer("num_total_batches", 10000, "Total number of batches to preprocess across all workers.")

def main(argv):
    del argv  # Unused.

    # --- Read environment variables for parallel execution ---
    # These are injected by the Kubernetes Job
    job_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
    parallelism = int(os.environ.get("PARALLELISM", 1))

    logging.info(f"--- Starting Preprocessing Worker {job_index + 1}/{parallelism} ---")

    # Each worker calculates its own subset of batches to process
    batches_per_worker = FLAGS.num_total_batches // parallelism
    start_batch = job_index * batches_per_worker
    end_batch = start_batch + batches_per_worker

    # The last worker picks up any remainder
    if job_index == parallelism - 1:
        end_batch = FLAGS.num_total_batches
    
    logging.info(f"This worker will process batches from {start_batch} to {end_batch - 1}")

    data_loader = CriteoDataLoader(
        file_pattern=FLAGS.source_data_path,
        global_batch_size=FLAGS.global_batch_size,
        is_training=False, 
    )
    data_iterator = data_loader.get_iterator()

    # Create the output directory on the local filesystem (GCS Fuse mount)
    os.makedirs(FLAGS.output_path, exist_ok=True)

    # Skip to the start batch for this worker
    for _ in range(start_batch):
        try:
            next(data_iterator)
        except StopIteration:
            logging.warning("Dataset ended before reaching start batch. Exiting.")
            return

    logging.info(f"Starting preprocessing...")
    for i in range(start_batch, end_batch):
        try:
            batch = next(data_iterator)
            
            output_file_path = os.path.join(FLAGS.output_path, f"batch_{i:05d}.npz")
            
            # Save directly to the local filesystem path
            with open(output_file_path, 'wb') as f:
                np.savez_compressed(f, **batch)

            if (i + 1) % 10 == 0:
                logging.info(f"Worker {job_index}: Successfully processed and saved batch {i + 1}")

        except StopIteration:
            logging.warning("Reached the end of the dataset before processing all requested batches.")
            break
            
    logging.info(f"--- Worker {job_index} Finished ---")

if __name__ == "__main__":
    app.run(main)
