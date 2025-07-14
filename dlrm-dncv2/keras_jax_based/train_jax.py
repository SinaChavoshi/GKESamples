import os
import time
import numpy as np
import tensorflow as tf

# Helper functions (constants, model/data creation) are defined first
# to be available in the global scope.
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128

def create_embedding_layer(keras_alias, per_replica_batch_size):
    """Creates the DistributedEmbedding layer."""
    feature_configs = {}
    for i, (vocab_size, multi_hot_size) in enumerate(zip(VOCAB_SIZES, MULTI_HOT_SIZES)):
        table = keras_alias.layers.TableConfig(
            name=f"table_{i}",
            vocabulary_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            optimizer=keras_alias.optimizers.Adagrad(learning_rate=0.025),
            placement="auto",
        )
        feature_configs[f"feature_{i}"] = keras_alias.layers.FeatureConfig(
            table=table,
            input_shape=(per_replica_batch_size, multi_hot_size),
        )
    return keras_alias.layers.DistributedEmbedding(
        feature_configs=feature_configs, name="distributed_embedding"
    )

def create_dlrm_model(keras_alias, embedding_layer):
    """Creates the full DLRM model."""
    dense_input = keras_alias.layers.Input(shape=(NUM_DENSE_FEATURES,), name="dense_features", dtype="float32")
    sparse_inputs = {
        f"feature_{i}": keras_alias.layers.Input(shape=(size,), name=f"feature_{i}", dtype="int32")
        for i, size in enumerate(MULTI_HOT_SIZES)
    }
    bottom_mlp = keras_alias.Sequential([
        keras_alias.layers.Dense(512, activation="relu"),
        keras_alias.layers.Dense(256, activation="relu"),
        keras_alias.layers.Dense(EMBEDDING_DIM, activation="relu")
    ], name="bottom_mlp")(dense_input)
    embedding_lookup = embedding_layer(sparse_inputs)
    embedding_vectors = list(embedding_lookup.values())
    all_features = keras_alias.layers.Concatenate(axis=1)(embedding_vectors + [bottom_mlp])
    interaction_output = tf.recommenders.layers.feature_interaction.MultiLayerDCN(
        num_layers=3, projection_dim=512, use_bias=True
    )(all_features)
    top_mlp_input = keras_alias.layers.Concatenate(axis=1)([bottom_mlp, interaction_output])
    top_mlp = keras_alias.Sequential([
        keras_alias.layers.Dense(1024, activation="relu"),
        keras_alias.layers.Dense(1024, activation="relu"),
        keras_alias.layers.Dense(512, activation="relu"),
        keras_alias.layers.Dense(256, activation="relu"),
        keras_alias.layers.Dense(1, activation="sigmoid")
    ], name="top_mlp")(top_mlp_input)
    return keras_alias.Model(inputs={"dense_features": dense_input, "sparse_features": sparse_inputs}, outputs=top_mlp)

def create_dataset(input_path, is_training, global_batch_size):
    """Creates a tf.data pipeline for Criteo data."""
    column_names = ['label'] + [f'int-feature-{i+1}' for i in range(NUM_DENSE_FEATURES)] + [f'categorical-feature-{i+1}' for i in range(len(VOCAB_SIZES))]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=input_path,
        batch_size=global_batch_size,
        column_names=column_names,
        label_name="label",
        header=False,
        field_delim='\t',
        num_epochs=None if is_training else 1,
        shuffle=is_training
    )
    def process_features(features, label):
        dense_features = tf.stack([tf.strings.to_number(features[f'int-feature-{i+1}'], out_type=tf.float32) for i in range(NUM_DENSE_FEATURES)], axis=1)
        sparse_features = {f"feature_{i}": tf.strings.to_number(features[f'categorical-feature-{i+1}'], out_type=tf.int32) for i in range(len(VOCAB_SIZES))}
        return {"dense_features": dense_features, "sparse_features": sparse_features}, label
    return dataset.map(process_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

def main():
    import jax
    import keras
    import keras_core as keras_rs
    
    class ThroughputLogger(keras.callbacks.Callback):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            self.total_examples = 0
        def on_train_batch_end(self, batch, logs=None):
            self.total_examples += self.batch_size
        def on_epoch_end(self, epoch, logs=None):
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - self.epoch_start_time
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
    print("--------------------------")
    
    # *** THE FIX IS HERE ***
    # For single-node runs, we don't need the Keras distribution strategy.
    # JAX's jit_compile will automatically use all TPU cores on the node.
    # We will only enable the distribution strategy for multi-node runs.
    distribution = None
    if os.environ.get("JAX_PROCESS_ID"):
        mesh = keras.distribution.DeviceMesh(
            shape=(len(devices),),
            axis_names=("batch",),
            devices=devices
        )
        distribution = keras.distribution.DataParallel(device_mesh=mesh)
        keras.distribution.set_distribution(distribution)
    
    GLOBAL_BATCH_SIZE = 32768
    PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // len(devices)

    model_dir = os.environ.get("MODEL_DIR", "/tmp/dlrm_jax_output")
    train_data_path = os.environ.get("TRAIN_DATA_PATH")
    eval_data_path = os.environ.get("EVAL_DATA_PATH")

    embedding_layer = create_embedding_layer(keras_rs, PER_REPLICA_BATCH_SIZE)
    model = create_dlrm_model(keras, embedding_layer)
    optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "auc"],
        jit_compile=True
    )
    model.summary()
    
    train_dataset = create_dataset(train_data_path, is_training=True, global_batch_size=GLOBAL_BATCH_SIZE)
    eval_dataset = create_dataset(eval_data_path, is_training=False, global_batch_size=GLOBAL_BATCH_SIZE)
    
    def jax_dataset_generator(dataset, is_training):
        for features, labels in dataset.as_numpy_iterator():
            preprocessed_sparse = embedding_layer.preprocess(features["sparse_features"], training=is_training)
            yield {"dense_features": features["dense_features"], "sparse_features": preprocessed_sparse}, labels
            
    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)
    
    model.fit(
        jax_dataset_generator(train_dataset, is_training=True),
        epochs=1,
        steps_per_epoch=100, # Using a smaller number for a quick test
        validation_data=jax_dataset_generator(eval_dataset, is_training=False),
        validation_steps=20,
        callbacks=[throughput_callback]
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()
