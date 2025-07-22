import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds

def get_dataset(batch_size, is_training=True):
    """Loads and prepares the MNIST dataset."""
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                              as_supervised=True, try_gcs=True)
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label
    dataset = dataset.map(scale)
    if is_training:
        dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def create_model():
    """Creates a simple CNN model."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

if __name__ == '__main__':
    print("Starting TPU Controller...")
    
    # --- TPU Initialization using the Controller-Worker Model ---
    jobset_name = os.environ['JOBSET_NAME']
    grpc_worker_name = os.environ['GRPC_WORKER_NAME']
    num_replicas = int(os.environ['NUM_REPLICAS'])

    # Build the gRPC endpoint string for the worker(s).
    endpoints = [f"grpc://{jobset_name}-{grpc_worker_name}-{i}-0.{jobset_name}:8470" for i in range(num_replicas)]
    tpu_address = ",".join(endpoints)
    print(f"Connecting to TPU worker(s) at: {tpu_address}")

    # Pass the explicit address to the resolver.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    # --- End of Initialization ---

    print(f"✅ TPU system initialized. Number of replicas: {strategy.num_replicas_in_sync}")

    PER_REPLICA_BATCH_SIZE = 128
    GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
    EPOCHS = 5
    STEPS_PER_EPOCH = 60000 // GLOBAL_BATCH_SIZE
    VALIDATION_STEPS = 10000 // GLOBAL_BATCH_SIZE

    train_dataset = get_dataset(GLOBAL_BATCH_SIZE, is_training=True)
    test_dataset = get_dataset(GLOBAL_BATCH_SIZE, is_training=False)

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy'],
            steps_per_execution=50
        )
    
    print("🚀 Starting model training...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_dataset,
        validation_steps=VALIDATION_STEPS
    )
    print("🎉 Training complete.")
