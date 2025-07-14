import os
import time
import jax
import keras
import numpy as np
import keras_core as keras_rs

from keras import layers

# --- Custom Callback and Model definitions can remain the same ---
# (ThroughputLogger, create_dlrm_model, etc.)

def load_preprocessed_data(data_dir):
    """Loads all data chunks from a directory and concatenates them."""
    print(f"Loading data from: {data_dir}")
    dense_chunks = []
    label_chunks = []
    sparse_chunks = [[] for _ in range(len(VOCAB_SIZES))]

    # Find all chunk files and load them
    files = sorted(os.listdir(data_dir))
    for f in files:
        if f.startswith('dense_chunk_'):
            dense_chunks.append(np.load(os.path.join(data_dir, f)))
        elif f.startswith('labels_chunk_'):
            label_chunks.append(np.load(os.path.join(data_dir, f)))
        elif f.startswith('sparse_chunk_'):
            parts = f.split('_')
            feature_index = int(parts[3].replace('.npy', ''))
            sparse_chunks[feature_index].append(np.load(os.path.join(data_dir, f)))

    # Concatenate all chunks
    dense_features = np.concatenate(dense_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    sparse_features = {}
    for i in range(len(VOCAB_SIZES)):
        sparse_features[f'feature_{i}'] = np.concatenate(sparse_chunks[i], axis=0)
    
    print(f"Successfully loaded {len(labels)} examples.")
    return {"dense_features": dense_features, "sparse_features": sparse_features}, labels


def main():
    import jax
    import keras
    
    jax.distributed.initialize()
    distribution = keras.distribution.DataParallel(devices=jax.devices("tpu"))
    keras.distribution.set_distribution(distribution)
    
    GLOBAL_BATCH_SIZE = 32768
    PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // jax.device_count()

    # --- Load pre-processed data from GCS FUSE mount point ---
    train_features, train_labels = load_preprocessed_data("/gcs/jax_preprocessed_data/train")
    eval_features, eval_labels = load_preprocessed_data("/gcs/jax_preprocessed_data/eval")

    with distribution.scope():
        embedding_layer = create_embedding_layer(PER_REPLICA_BATCH_SIZE)
        model = create_dlrm_model(embedding_layer)
        optimizer = keras.optimizers.Adagrad(learning_rate=0.00025)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"], jit_compile=True)
    
    model.summary()
    
    # Preprocess the sparse features once, outside the training loop
    preprocessed_train_sparse = embedding_layer.preprocess(train_features["sparse_features"])
    preprocessed_eval_sparse = embedding_layer.preprocess(eval_features["sparse_features"])

    train_data = {"dense_features": train_features["dense_features"], "sparse_features": preprocessed_train_sparse}
    eval_data = {"dense_features": eval_features["dense_features"], "sparse_features": preprocessed_eval_sparse}
    
    throughput_callback = ThroughputLogger(batch_size=GLOBAL_BATCH_SIZE)
    model.fit(
        train_data,
        train_labels,
        batch_size=GLOBAL_BATCH_SIZE,
        epochs=1,
        steps_per_epoch=10000,
        validation_data=(eval_data, eval_labels),
        callbacks=[throughput_callback]
    )
    
    print("Training complete.")

# --- Make sure all helper functions are defined before main() is called ---
if __name__ == "__main__":
    # Define constants and helper functions here...
    main()
