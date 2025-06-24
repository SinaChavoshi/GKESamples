from collections.abc import Sequence
import os

from absl import app

# Set the Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import tensorflow as tf
import keras
import keras_rs

keras.config.disable_traceback_filtering()

# --- NEW: Configuration for the real dataset ---
GCS_BUCKET = "gke-training-sample"
DATA_DIR = f"gs://{GCS_BUCKET}/movielens"
CSV_HEADER = ["user_id", "movie_id", "user_rating", "timestamp"]
CSV_COLUMN_DEFAULTS = ["", "", 0.0, 0]

# --- Model definition remains the same ---
class EmbeddingModel(keras.Model):
    def __init__(self, feature_configs):
        super().__init__()
        self.embedding_layer = keras_rs.layers.DistributedEmbedding(
            feature_configs=feature_configs
        )
        self.ratings = keras.Sequential([
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ])

    def call(self, preprocessed_features):
        embeddings = self.embedding_layer(preprocessed_features)
        return self.ratings(
            keras.ops.concatenate(
                [embeddings["user_id"], embeddings["movie_id"]], axis=1
            )
        )

# --- NEW: Data processing function for the real dataset ---
def get_dataset(file_pattern, batch_size):
    """Creates a sharded tf.data.Dataset from CSV files in GCS."""
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=CSV_COLUMN_DEFAULTS,
        header=False,
        field_delim="\t",
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=100_000,
        shuffle_seed=42,
    )

    # --- CRITICAL FOR DISTRIBUTED TRAINING ---
    # Shard the dataset so each host gets a unique slice of the data.
    # This prevents the 4 nodes from doing redundant work.
    dataset = dataset.shard(
        num_shards=jax.process_count(), index=jax.process_index()
    )

    def process_features(features):
        # The label is the 'user_rating'
        labels = features.pop("user_rating")
        # The other columns are model features
        return features, labels

    return dataset.map(process_features, num_parallel_calls=tf.data.AUTOTUNE)

def main(argv: Sequence[str]) -> None:
    print(f"DEVICES: {jax.devices()}")
    print(f"BACKEND: {keras.backend.backend()}")
    print(f"Total Hosts: {jax.process_count()}, My Host ID: {jax.process_index()}")

    # Dimension configuration
    batch_size = 256
    movie_vocabulary_size = 2048
    movie_embedding_size = 64
    user_vocabulary_size = 2048
    user_embedding_size = 64

    # --- REPLACED: Load data from GCS instead of the fake generator ---
    # Create sharded datasets for training and testing
    full_dataset = get_dataset(f"{DATA_DIR}/*", batch_size)

    # Note: For simplicity, we use the same data for train/test.
    # In a real scenario, you'd have separate train/test files.
    train_dataset = full_dataset
    test_dataset = full_dataset.take(10) # Take a small subset for validation

    # --- Embedding layer configurations remain the same ---
    movie_table = keras_rs.layers.TableConfig(
        name="movie_table",
        vocabulary_size=movie_vocabulary_size,
        embedding_dim=movie_embedding_size,
    )
    user_table = keras_rs.layers.TableConfig(
        name="user_table",
        vocabulary_size=user_vocabulary_size,
        embedding_dim=user_embedding_size,
    )
    feature_config = {
        "movie_id": keras_rs.layers.FeatureConfig(
            name="movie", table=movie_table
        ),
        "user_id": keras_rs.layers.FeatureConfig(
            name="user", table=user_table
        ),
    }

    # Set up distribution strategy
    keras.distribution.set_distribution(keras.distribution.DataParallel())

    # Construct and compile the model
    model = EmbeddingModel(feature_config)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
        optimizer="adagrad",
    )
    
    # --- SIMPLIFIED: Preprocessing is now part of the dataset pipeline ---
    # The `keras_rs` layer's preprocessing is automatically handled by Keras 3
    # when the model is compiled within a distribution scope.

    # Train the model
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        validation_freq=5,
        epochs=50,
    )

if __name__ == "__main__":
    app.run(main)
