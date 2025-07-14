import os
#
# JAX and Keras imports are now inside main()
#
import numpy as np
import tensorflow as tf
from keras import layers
import keras_core as keras_rs

# --- Model Configuration (can stay at the global level) ---
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
NUM_DENSE_FEATURES = 13
EMBEDDING_DIM = 128
GLOBAL_BATCH_SIZE = 32768

# ... (create_embedding_layer, create_dlrm_model, and create_dataset functions remain the same) ...

# --- Main Training Function ---
def main():
    # --- JAX and Keras Initialization (Moved Inside Main) ---
    import jax
    import keras

    print("--- JAX Initialization ---")
    # This will now correctly initialize the multi-host environment
    jax.distributed.initialize()
    print(f"Global JAX devices: {jax.devices()}")
    print(f"Local JAX devices: {jax.local_devices()}")
    print("--------------------------")

    # This remains the same
    distribution = keras.distribution.DataParallel(devices=jax.devices("tpu"))
    keras.distribution.set_distribution(distribution)
    
    # Calculate per-replica batch size *after* JAX initialization
    PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // jax.device_count()

    model_dir = os.environ.get("MODEL_DIR", "/tmp/dlrm_jax_output")
    train_data_path = os.environ.get("TRAIN_DATA_PATH")
    eval_data_path = os.environ.get("EVAL_DATA_PATH")

    print(f"Model directory: {model_dir}")
    print(f"Train data path: {train_data_path}")
    print(f"Eval data path: {eval_data_path}")

    # Create the model under the distribution scope
    with distribution.scope():
        embedding_layer = create_embedding_layer()
        model = create_dlrm_model(embedding_layer)
        optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "auc"],
            jit_compile=True
        )
    model.summary()

    # Create the datasets
    train_dataset = create_dataset(train_data_path, is_training=True)
    eval_dataset = create_dataset(eval_data_path, is_training=False)

    # Define a preprocessing generator for JAX
    def jax_train_dataset_generator():
        for features, labels in train_dataset.as_numpy_iterator():
            preprocessed_sparse = embedding_layer.preprocess(features["sparse_features"])
            yield {"dense_features": features["dense_features"], "sparse_features": preprocessed_sparse}, labels

    # Train the model
    model.fit(
        jax_train_dataset_generator(),
        epochs=1,
        steps_per_epoch=10000,
        validation_data=eval_dataset,
        validation_steps=100
    )

    # Save the model
    model.save(os.path.join(model_dir, "final_model.keras"))
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
