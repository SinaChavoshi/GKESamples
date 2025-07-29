# main.py
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
    # TFJob automatically sets the TF_CONFIG environment variable.
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_type = tf_config.get('task', {}).get('type', '')
    task_id = tf_config.get('task', {}).get('index', 0)
    print(f"Starting TFJob replica. Role: {task_type}, ID: {task_id}")
    print(f"Full TF_CONFIG: {os.environ.get('TF_CONFIG')}")

    # This resolver reads the TF_CONFIG environment variable.
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    
    # For TPUStrategy, the initialization steps remain the same.
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
    
    # Only the chief/master replica should run the training loop.
    if task_type == 'master' or (task_type == 'worker' and task_id == 0):
      print("🚀 This replica is the chief. Starting model training...")
      model.fit(
          train_dataset,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=test_dataset,
          validation_steps=VALIDATION_STEPS
      )
      print("🎉 Training complete.")
    else:
      # Other workers just join the cluster and wait for instructions.
      print("This is a worker replica. Joining cluster...")
      resolver.cluster_spec().as_cluster_def() # This keeps the worker alive
