import os
import tensorflow as tf
import tensorflow_datasets as tfds

def print_tpu_environment_variables():
    """Prints environment variables relevant for TPUClusterResolver."""
    print("--- Verifying TPU Environment Variables ---")
    
    tpu_worker_hostnames = os.environ.get('TPU_WORKER_HOSTNAMES')
    tpu_worker_id = os.environ.get('TPU_WORKER_ID')
    tf_config = os.environ.get('TF_CONFIG')

    print(f"TPU_WORKER_HOSTNAMES: {tpu_worker_hostnames}")
    print(f"TPU_WORKER_ID: {tpu_worker_id}")
    
    if tf_config:
        print(f"TF_CONFIG: {tf_config}")
    else:
        # This is the expected outcome for TPUStrategy on GKE.
        print("TF_CONFIG: Not set (as expected for this setup).")
    
    print("------------------------------------------")


def get_dataset(batch_size, is_training=True):
    """Loads and prepares the MNIST dataset from a public GCS bucket."""
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                              as_supervised=True, try_gcs=True)

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    dataset = dataset.map(scale)

    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

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
    # Print the environment variables for verification.
    print_tpu_environment_variables()
    
    print("Initializing TPU environment...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.from_environ()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

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
