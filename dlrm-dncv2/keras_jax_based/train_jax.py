import os
import time
import numpy as np
import tensorflow as tf
import keras

# --- Custom DCN Layer to remove tfrs dependency ---
class CustomMultiLayerDCN(keras.layers.Layer):
    """A custom implementation of the Multi-Layer DCN cross network."""
    def __init__(self, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        # The dense layers will be created in build() once the input shape is known.
        self._dense_layers = []

    def build(self, input_shape):
        # Create the dense layers with the correct output dimension, which is the
        # same as the input dimension for the cross network.
        dim = input_shape[-1]
        self._dense_layers = [
            keras.layers.Dense(dim, use_bias=True) for _ in range(self.num_layers)
        ]
        super().build(input_shape)

    def call(self, x0):
        x = x0
        for i in range(self.num_layers):
            # The core DCN logic: x_{i+1} = x_0 * (W * x_i + b) + x_i
            x_proj = self._dense_layers[i](x)
            x = x0 * x_proj + x
        return x

# Helper functions (constants, model/data creation) are defined first
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128

def create_embedding_layer(per_replica_batch_size):
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
    """Creates the full DLRM model."""
    dense_input = keras.layers.Input(shape=(NUM_DENSE_FEATURES,), name="dense_features", dtype="float32")
    sparse_inputs = {
        f"feature_{i}": keras.layers.Input(shape=(size,), name=f"feature_{i}", dtype="int32")
        for i, size in enumerate(MULTI_HOT_SIZES)
    }
    bottom_mlp = keras.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(EMBEDDING_DIM, activation="relu")
    ], name="bottom_mlp")(dense_input)
    
    # Process sparse inputs with their respective embedding layers
    embedding_vectors = []
    for i, emb_layer in enumerate(embedding_layers):
        lookup = emb_layer(sparse_inputs[f"feature_{i}"])
        # Wrap the tf.reduce_mean operation in a Lambda layer
        mean_embedding = keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1)
        )(lookup)
        embedding_vectors.append(mean_embedding)

    all_features = keras.layers.Concatenate(axis=1)(embedding_vectors + [bottom_mlp])
    
    # Use the new custom DCN layer
    interaction_output = CustomMultiLayerDCN(
        num_layers=3
    )(all_features)

    top_mlp_input = keras.layers.Concatenate(axis=1)([bottom_mlp, interaction_output])
    top_mlp = keras.Sequential([
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ], name="top_mlp")(top_mlp_input)
    return keras.Model(inputs={"dense_features": dense_input, "sparse_features": sparse_inputs}, outputs=top_mlp)

def create_dataset(input_path, is_training, global_batch_size):
    """Creates a tf.data pipeline for the synthetic Criteo data."""
    # Define the defaults for each column to enforce the correct data types.
    # Label and dense features are floats.
    dense_defaults = [tf.constant([0.0], dtype=tf.float32)] * (1 + NUM_DENSE_FEATURES)
    # Categorical features must be read as strings.
    categorical_defaults = [tf.constant(["0"], dtype=tf.string)] * len(VOCAB_SIZES)
    column_defaults = dense_defaults + categorical_defaults
    
    column_names = ['label'] + [f'int-feature-{i+1}' for i in range(NUM_DENSE_FEATURES)] + [f'categorical-feature-{i+1}' for i in range(len(VOCAB_SIZES))]
    
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=input_path,
        batch_size=global_batch_size,
        column_names=column_names,
        column_defaults=column_defaults,  # Use the defined defaults
        label_name="label",
        header=False,
        field_delim='\t',
        num_epochs=None if is_training else 1,
        shuffle=is_training
    )
    
    def process_features(features, label):
        # Dense features are already floats, no conversion needed.
        dense_features = tf.stack([features[f'int-feature-{i+1}'] for i in range(NUM_DENSE_FEATURES)], axis=1)
        
        sparse_features_processed = {}
        for i, size in enumerate(MULTI_HOT_SIZES):
             # Categorical features are now strings, so we convert them to numbers.
             cat_feature = tf.strings.to_number(features[f'categorical-feature-{i+1}'], out_type=tf.int32)
             # Tile the features for multi-hot representation.
             sparse_features_processed[f"feature_{i}"] = tf.tile(tf.expand_dims(cat_feature, -1), [1, size])
             
        return {"dense_features": dense_features, "sparse_features": sparse_features_processed}, label
        
    return dataset.map(process_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

def main():
    import jax
    
    class ThroughputLogger(keras.callbacks.Callback):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size
            self.total_examples = 0 # Initialize here
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            self.total_examples = 0 # Reset for each epoch
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
        mesh = keras.distribution.DeviceMesh(
            shape=(len(devices),),
            axis_names=("batch",),
            devices=devices
        )
        distribution = keras.distribution.DataParallel(device_mesh=mesh)
        keras.distribution.set_distribution(distribution)
        print("Keras distribution set for JAX DataParallel.")

    GLOBAL_BATCH_SIZE = 32768
    PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // len(devices) if devices else GLOBAL_BATCH_SIZE

    embedding_layers = create_embedding_layer(PER_REPLICA_BATCH_SIZE)
    model = create_dlrm_model(embedding_layers)
    optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "auc"],
        jit_compile=True
    )
    model.summary()
    
    # Use environment variables set in the JobSet to find the data
    train_data_path = os.environ.get("TRAIN_DATA_PATH", "/gcs/synthetic_data/train/*")
    eval_data_path = os.environ.get("EVAL_DATA_PATH", "/gcs/synthetic_data/eval/*")
    
    train_dataset = create_dataset(train_data_path, is_training=True, global_batch_size=GLOBAL_BATCH_SIZE)
    eval_dataset = create_dataset(eval_data_path, is_training=False, global_batch_size=GLOBAL_BATCH_SIZE)
            
    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)
    
    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=100,
        validation_data=eval_dataset,
        validation_steps=20,
        callbacks=[throughput_callback]
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()
