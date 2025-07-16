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
NUM_SPARSE_FEATURES = len(VOCAB_SIZES)
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128
DCN_LOW_RANK_DIM = 512
DCN_NUM_LAYERS = 3

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
    """Creates the full DLRM model"""
    dense_input = keras.layers.Input(shape=(NUM_DENSE_FEATURES,), name="dense_features", dtype="float32")
    sparse_inputs = {}
    # Model inputs are named feature_0, feature_1, etc.
    for i in range(NUM_SPARSE_FEATURES):
        sparse_inputs[f"feature_{i}"] = keras.layers.Input(shape=(None,), name=f"feature_{i}", dtype="int64", ragged=True)

    bottom_mlp = keras.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(EMBEDDING_DIM, activation="relu")
    ], name="bottom_mlp")(dense_input)

    embedding_vectors = []
    for i, emb_layer in enumerate(embedding_layers):
        lookup = emb_layer(sparse_inputs[f"feature_{i}"])
        mean_embedding = keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1)
        )(lookup)
        embedding_vectors.append(mean_embedding)

    x0 = keras.layers.Concatenate(axis=1)(embedding_vectors + [bottom_mlp])

    x = x0
    for _ in range(DCN_NUM_LAYERS):
        x = keras_rs.layers.FeatureCross(
            projection_dim=DCN_LOW_RANK_DIM,
            use_bias=True,
        )(x0, x)
    interaction_output = x

    top_mlp_input = keras.layers.Concatenate(axis=1)([bottom_mlp, interaction_output])

    top_mlp = keras.Sequential([
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ], name="top_mlp")(top_mlp_input)

    return keras.Model(inputs={"dense_features": dense_input, "sparse_features": sparse_inputs}, outputs=top_mlp)

def create_dataset_from_tfrecords(input_path, is_training, global_batch_size):
    """Creates a tf.data pipeline from TFRecord files using the correct feature names."""
    feature_spec = {
        'label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=None)
    }
    for i in range(1, NUM_DENSE_FEATURES + 1):
        feature_spec[f'dense-feature-{i}'] = tf.io.FixedLenFeature(
            [1], dtype=tf.float32, default_value=None
        )
    for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
        feature_spec[f'sparse-feature-{i}'] = tf.io.VarLenFeature(dtype=tf.int64)

    def _parse_fn(features):
        """Parses a single tf.train.Example."""
        label = tf.cast(features.pop('label'), tf.float32)
        
        dense_features_list = []
        for i in range(1, NUM_DENSE_FEATURES + 1):
            dense_feat = features[f'dense-feature-{i}']
            dense_features_list.append(dense_feat)
        dense_features = tf.concat(dense_features_list, axis=1)

        sparse_features = {}
        for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
            model_feature_index = i - (NUM_DENSE_FEATURES + 1)
            sparse_features[f"feature_{model_feature_index}"] = tf.sparse.to_dense(features[f'sparse-feature-{i}'])
            
        return {"dense_features": dense_features, "sparse_features": sparse_features}, label

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
    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)

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
        distribution = keras.distribution.DataParallel(devices=jax.devices())
        keras.distribution.set_distribution(distribution)
        print("Keras distribution set for JAX DataParallel.")
    else:
        print("--- Running in Single-Host Mode ---")
        distribution = None
    devices = jax.devices()
    print(f"Global JAX devices: {devices}")

    GLOBAL_BATCH_SIZE = 32768
    
    num_replicas = jax.device_count()
    if num_replicas > 0:
        per_replica_batch_size = GLOBAL_BATCH_SIZE // num_replicas
    else:
        per_replica_batch_size = GLOBAL_BATCH_SIZE

    print("--- Batch Size Configuration ---")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"Number of replicas: {num_replicas}")
    print(f"Per-replica batch size: {per_replica_batch_size}")

    embedding_layers = create_embedding_layer()
    model = create_dlrm_model(embedding_layers)
    optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"], jit_compile=True)
    model.summary()

    train_data_path = os.environ.get("TRAIN_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*")
    eval_data_path = os.environ.get("EVAL_DATA_PATH", "gs://zyc_dlrm/dataset/tb_tf_record_train_val/eval/day_*/*")

    # --- Manually create and distribute the dataset as per https://github.com/keras-team/keras-rs/pull/131
    train_dataset_raw = create_dataset_from_tfrecords(train_data_path, is_training=True, global_batch_size=GLOBAL_BATCH_SIZE)
    eval_dataset_raw = create_dataset_from_tfrecords(eval_data_path, is_training=False, global_batch_size=GLOBAL_BATCH_SIZE)

    if distribution:
        print("--- Manually sharding dataset for multi-host training ---")
        train_dataset = distribution.distribute_dataset(train_dataset_raw)
        eval_dataset = distribution.distribute_dataset(eval_dataset_raw)
    else:
        train_dataset = train_dataset_raw
        eval_dataset = eval_dataset_raw

    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)

    train_steps = 2
    validation_steps = 1

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=validation_steps,
        callbacks=[throughput_callback]
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
