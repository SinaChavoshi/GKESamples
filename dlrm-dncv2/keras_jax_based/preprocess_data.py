import os
import argparse
import numpy as np
import tensorflow as tf
from google.cloud import storage

# --- Constants ---
NUM_DENSE_FEATURES = 13
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
COLUMN_NAMES = (['label'] +
                [f'int-feature-{i+1}' for i in range(NUM_DENSE_FEATURES)] +
                [f'categorical-feature-{i+1}' for i in range(len(VOCAB_SIZES))])

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def preprocess_and_upload(input_path, bucket_name, gcs_prefix):
    """Reads raw data, preprocesses it, and uploads to GCS."""
    print(f"--- Starting preprocessing for {input_path} ---")
    local_temp_dir = "temp_data"
    if not os.path.exists(local_temp_dir):
        os.makedirs(local_temp_dir)

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=input_path,
        batch_size=8192, # Use a larger batch size for efficiency
        column_names=COLUMN_NAMES,
        label_name='label',
        header=False,
        field_delim='\t',
        num_epochs=1,
        shuffle=False
    )

    # Process data in chunks
    for i, (features, labels) in enumerate(dataset):
        print(f"Processing chunk {i}...")
        dense_features = tf.stack([tf.strings.to_number(features[f'int-feature-{j+1}'], out_type=tf.float32) for j in range(NUM_DENSE_FEATURES)], axis=1)
        sparse_features = [tf.strings.to_number(features[f'categorical-feature-{k+1}'], out_type=tf.int32) for k in range(len(VOCAB_SIZES))]

        # Save chunk locally first
        np.save(os.path.join(local_temp_dir, f'dense_chunk_{i}.npy'), dense_features.numpy())
        np.save(os.path.join(local_temp_dir, f'labels_chunk_{i}.npy'), labels.numpy())
        for k in range(len(VOCAB_SIZES)):
            np.save(os.path.join(local_temp_dir, f'sparse_chunk_{i}_feature_{k}.npy'), sparse_features[k].numpy())

    print("--- Uploading processed chunks to GCS ---")
    for file_name in os.listdir(local_temp_dir):
        destination_path = f"{gcs_prefix}/{file_name}"
        upload_blob(bucket_name, os.path.join(local_temp_dir, file_name), destination_path)

    # Clean up local temp files
    for file_name in os.listdir(local_temp_dir):
        os.remove(os.path.join(local_temp_dir, file_name))
    os.rmdir(local_temp_dir)
    print("--- Preprocessing and upload complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", required=True, help="Your GCS bucket name.")
    args = parser.parse_args()

    # Process training data
    preprocess_and_upload(
        input_path="gs://criteo-tpu-us-east5/criteo_preprocessed_shuffled_unbatched/train/*",
        bucket_name=args.bucket_name,
        gcs_prefix="jax_preprocessed_data/train"
    )

    # Process evaluation data
    preprocess_and_upload(
        input_path="gs://criteo-tpu-us-east5/criteo_preprocessed_shuffled_unbatched/eval/*",
        bucket_name=args.bucket_name,
        gcs_prefix="jax_preprocessed_data/eval"
    )
