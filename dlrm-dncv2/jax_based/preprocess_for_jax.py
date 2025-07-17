import os
import tempfile
import shutil
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

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
flags.DEFINE_integer("processing_batch_size", 5000, "Batch size for processing records.")
flags.DEFINE_integer("num_total_batches", 10000, "Total number of batches to preprocess across all workers.")

def main(argv):
    del argv  # Unused.

    # --- Parallel processing logic ---
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

    # Ensure the output directory on the fuse mount exists
    os.makedirs(FLAGS.output_path, exist_ok=True)

    logging.info("Starting preprocessing loop...")
    # The loop now iterates over the specific slice for this worker
    for i in range(start_batch, end_batch):
        try:
            batch = next(data_iterator)

            # Flatten the batch structure for saving
            numpy_batch = {}
            for key, value in batch.items():
                if key == 'sparse_features' and isinstance(value, dict):
                    for sparse_key, sparse_tensor in value.items():
                        numpy_batch[f'sparse_{sparse_key}'] = sparse_tensor.numpy()
                elif isinstance(value, tf.Tensor):
                    numpy_batch[key] = value.numpy()

            # Define temporary and final file paths.
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"batch_{i:05d}.npz.tmp")
            final_output_path = os.path.join(FLAGS.output_path, f"batch_{i:05d}.npz")
            
            # --- Enhanced Error Handling ---
            # 1. Try to save the file to a local temporary path.
            try:
                np.savez_compressed(temp_file_path, **numpy_batch)
            except Exception as e:
                logging.error(f"CRITICAL: np.savez_compressed FAILED for batch {i}. Error: {e}")
                # Log the data types to debug data mismatch issues.
                for k, v in numpy_batch.items():
                    logging.error(f"Batch {i} key '{k}' has dtype: {getattr(v, 'dtype', 'N/A')} and shape: {getattr(v, 'shape', 'N/A')}")
                continue  # Skip this problematic batch and move to the next.

            # 2. If saving was successful, try to copy the file.
            try:
                shutil.copy(temp_file_path, final_output_path)
            except Exception as e:
                logging.error(f"Failed to copy batch {i} from temp storage. Error: {e}")
                continue # Skip if copy fails
            finally:
                # 3. Always clean up the temporary file if it exists.
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)


            if (i + 1) % 100 == 0:
                logging.info(f"Worker {job_index}: Successfully processed and saved batch {i + 1} to {final_output_path}")

        except StopIteration:
            logging.warning("Reached the end of the dataset before processing all requested batches.")
            break

    logging.info(f"--- Worker {job_index} Finished ---")

if __name__ == "__main__":
    app.run(main)
