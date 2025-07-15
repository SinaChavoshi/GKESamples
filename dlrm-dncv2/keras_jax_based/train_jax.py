import os
import time
import numpy as np
import tensorflow as tf
import keras
import jax
# Make sure to install keras-rs: pip install keras-rs
import keras_rs

# Constants remain the same
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128
DCN_LOW_RANK_DIM = 512
DCN_NUM_LAYERS = 3

def create_dataset_from_tfrecords(input_path, is_training, global_batch_size):
    """Creates a tf.data pipeline from TFRecord files."""
    feature_spec = {
        'label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=None)
    }
    # Use 1-based indexing for feature names to match benchmark data
    for i in range(NUM_DENSE_FEATURES):
        feature_spec[f'int-feature-{i+1}'] = tf.io.FixedLenFeature(
            [1], dtype=tf.int64, default_value=None
        )
    for i in range(len(VOCAB_SIZES)):
         feature_spec[f'cat-feature-{i+1}'] = tf.io.VarLenFeature(dtype=tf.int64)

    dataset = tf.data.Dataset.list_files(input_path, shuffle=is_training)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )
    dataset = dataset.batch(global_batch_size, drop_remainder=is_training)
    dataset = dataset.map(
        lambda x: tf.io.parse_example(x, feature_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _parse_fn(features):
        # Cast the label from int64 to float32
        label = tf.cast(features.pop('label'), tf.float32)
        
        dense_features_list = []
        # Use 1-based keys to access parsed dense features
        for i in range(NUM_DENSE_FEATURES):
            dense_feat = tf.cast(features[f'int-feature-{i+1}'], tf.float32)
            dense_features_list.append(dense_feat)
        dense_features = tf.concat(dense_features_list, axis=1)

        sparse_features = {}
        # Use 1-based keys for cat features, map to 0-based for model
        for i in range(len(VOCAB_SIZES)):
            sparse_features[f"feature_{i}"] = tf.sparse.to_dense(features[f'cat-feature-{i+1}'])
            
        return {"dense_features": dense_features, "sparse_features": sparse_features}, label

    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def create_embedding_layer():
    """Creates the embedding layers using standard Keras."""
    embedding_layers = []
    for i, vocab_size in enumerate(VOCAB_SIZES):
        embedding_layers.append(
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBEDDING_DIM,
                name=f"embedding_{i}"
            )
        )
    return embedding_layers

def create_dlrm_model(embedding_layers):
    """Creates the full DLRM model-DNCv2 model"""
    dense_input = keras.layers.Input(shape=(NUM_DENSE_FEATURES,), name="dense_features", dtype="float32")
    sparse_inputs = {}
    for i in range(len(VOCAB_SIZES)):
        sparse_inputs[f"feature_{i}"] = keras.layers.Input(shape=(None,), name=f"feature_{i}", dtype="int64", ragged=True)

    bottom_mlp = keras.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(EMBEDDING_DIM, activation="relu")
    ], name="bottom_mlp")(dense_input)

    embedding_vectors = []
    for i, emb_layer in enumerate(embedding_layers):
        lookup = emb_layer(sparse_inputs[f"feature_{i}"])
        mean_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(lookup)
        embedding_vectors.append(mean_embedding)

    # Concatenate all features to form the input for the cross network
    x0 = keras.layers.Concatenate(axis=1)(embedding_vectors + [bottom_mlp])

    # Build the DCN-V2 Cross Network using the FeatureCross layer
    x = x0
    for _ in range(DCN_NUM_LAYERS):
        x = keras_rs.layers.FeatureCross(
            projection_dim=DCN_LOW_RANK_DIM, # This enables DCN-V2 logic
            use_bias=True,
        )(x0, x)
    interaction_output = x

    top_mlp_input = keras.layers.Concatenate(axis=1)([bottom_mlp, interaction_output])

    # Top MLP (matches benchmark: [1024, 1024, 512, 256, 1])
    top_mlp = keras.Sequential([
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ], name="top_mlp")(top_mlp_input)

    return keras.Model(inputs={"dense_features": dense_input, "sparse_features": sparse_inputs}, outputs=top_mlp)

def main():
    class ThroughputLogger(keras.callbacks.Callback):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size
            self.total_examples = 0
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            self.total_examples = 0
        def on_train_batch_end(self, batch, logs=None):
            self.total_examples += self.batch_size
        def on_epoch_end(self, epoch, logs=None):
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - self.epoch_start_time
            if elapsed_time > 0:
                throughput = self.total_examples / elapsed_time
                print(f"\nEpoch {epoch + 1} - Throughput: {throughput:.2f} examples/sec")
                if logs is not None:
                    logs['throughput'] = throughput

    if os.environ.get("JAX_PROCESS_ID"):
        print("--- Initializing JAX for Multi-Host Environment ---")
        jax.distributed.initialize()
    else:
        print("--- Running in Single-Host Mode ---")

    devices = jax.devices()
    print(f"Global JAX devices: {devices}")

    if os.environ.get("JAX_PROCESS_ID"):
        mesh = keras.distribution.DeviceMesh(shape=(len(devices),), axis_names=("batch",), devices=devices)
        distribution = keras.distribution.DataParallel(device_mesh=mesh)
        keras.distribution.set_distribution(distribution)
        print("Keras distribution set for JAX DataParallel.")

    GLOBAL_BATCH_SIZE = 32768
    
    # --- Calculate and log the per-replica batch size for clarity ---
    num_replicas = jax.device_count()
    if num_replicas > 0:
        per_replica_batch_size = GLOBAL_BATCH_SIZE // num_replicas
    else:
        # Fallback for CPU-only environments (no devices found)
        per_replica_batch_size = GLOBAL_BATCH_SIZE

    print("--- Batch Size Configuration ---")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"Number of replicas: {num_replicas}")
    print(f"Per-replica batch size: {per_replica_batch_size}")
    # --- End of batch size logging ---

    embedding_layers = create_embedding_layer()
    model = create_dlrm_model(embedding_layers)
    optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"], jit_compile=True)
    model.summary()

    train_data_path = os.environ.get("TRAIN_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*")
    eval_data_path = os.environ.get("EVAL_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/eval/day_*/*")

    train_dataset = create_dataset_from_tfrecords(train_data_path, is_training=True, global_batch_size=GLOBAL_BATCH_SIZE)
    eval_dataset = create_dataset_from_tfrecords(eval_data_path, is_training=False, global_batch_size=GLOBAL_BATCH_SIZE)

    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)

    # Updated training and validation steps to match the benchmark
    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=10000,
        validation_data=eval_dataset,
        validation_steps=660,
        callbacks=[throughput_callback]
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
