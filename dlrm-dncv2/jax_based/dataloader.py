import os
from absl import app
from absl import flags
from absl import logging
import gcsfs
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
    "gs://your-bucket-name/dlrm_preprocessed_jax/train", # <--- UPDATE THIS
    "GCS path to save the preprocessed NumPy arrays."
)
flags.DEFINE_integer("global_batch_size", 32768, "Global batch size for processing.")
flags.DEFINE_integer("num_batches_to_process", 100, "Number of batches to preprocess and save.")

def main(argv):
    del argv  # Unused.

    logging.info("Setting up Criteo data loader to read TFRecords...")
    data_loader = CriteoDataLoader(
        file_pattern=FLAGS.source_data_path,
        global_batch_size=FLAGS.global_batch_size,
        is_training=False,  # Set to False to read sequentially
    )
    data_iterator = data_loader.get_iterator()

    # GCS filesystem object to save files directly
    gcs = gcsfs.GCSFileSystem()

    logging.info(f"Starting preprocessing for {FLAGS.num_batches_to_process} batches...")
    for i in range(FLAGS.num_batches_to_process):
        try:
            batch = next(data_iterator)
            
            # The batch is a dictionary of NumPy arrays, ready to be saved.
            # We save each batch as a separate compressed .npz file.
            output_file_path = os.path.join(FLAGS.output_path, f"batch_{i:04d}.npz")
            
            with gcs.open(output_file_path, 'wb') as f:
                np.savez_compressed(f, **batch)

            if (i + 1) % 10 == 0:
                logging.info(f"Successfully processed and saved batch {i + 1}/{FLAGS.num_batches_to_process}")

        except StopIteration:
            logging.warning("Reached the end of the dataset before processing all requested batches.")
            break
            
    logging.info("Preprocessing complete.")

if __name__ == "__main__":

    app.run(main)

