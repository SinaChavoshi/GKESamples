from collections.abc import Sequence
import os

from absl import app

os.environ["KERAS_BACKEND"] = "jax"

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_rs

keras.config.disable_traceback_filtering()


# Define a ranking model with embedding layer.
class EmbeddingModel(keras.Model):

  def __init__(self, feature_configs):
    super().__init__()

    self.embedding_layer = keras_rs.layers.DistributedEmbedding(
        feature_configs=feature_configs
    )
    self.ratings = keras.Sequential([
        # Learn multiple dense layers.
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        keras.layers.Dense(1),
    ])

  def call(self, preprocessed_features):
    embeddings = self.embedding_layer(preprocessed_features)
    return self.ratings(
        keras.ops.concatenate(
            [embeddings["user_id"], embeddings["movie_id"]], axis=1
        )
    )


def fake_movielens_generator(size: int = 100000):
  for _ in range(size):
    yield {
        "movie_id": tf.strings.as_string(
            tf.random.uniform(
                shape=tuple(), dtype=tf.int32, minval=0, maxval=2048
            )
        ),
        "user_id": tf.strings.as_string(
            tf.random.uniform(
                shape=tuple(), dtype=tf.int32, minval=0, maxval=2048
            )
        ),
        "user_rating": (
            tf.cast(
                tf.random.uniform(
                    shape=tuple(), minval=0, maxval=10, dtype=tf.int32
                ),
                tf.float32,
            )
            / 2.0
        ),
    }


def main(argv: Sequence[str]) -> None:

  print("Devices:", jax.devices())
  print("Backend:", keras.backend.backend())
  print("Keras version:", keras.__version__)
  print("Keras RS version:", keras_rs.__version__)
  print(f"Total Hosts: {jax.process_count()}, My Host ID: {jax.process_index()}")


  # Ratings data.
  # tfds.load("movielens/100k-ratings", split="train")
  # Fake the data for use in a test.  The structure matches that which comes
  # directly out of TFDS.
  ratings = tf.data.Dataset.from_generator(
      fake_movielens_generator,
      args=[100000],
      output_signature={
          "movie_id": tf.TensorSpec(shape=(), dtype=tf.string),
          "user_id": tf.TensorSpec(shape=(), dtype=tf.string),
          "user_rating": tf.TensorSpec(shape=(), dtype=tf.float32),
      },
  )

  # Select the basic features.
  ratings = ratings.map(
      lambda x: (
          {
              "movie_id": tf.strings.to_number(x["movie_id"]),
              "user_id": tf.strings.to_number(x["user_id"]),
          },
          x["user_rating"],
      )
  )

  # Dimension configuration.
  batch_size = 256
  movie_vocabulary_size = 2048
  movie_embedding_size = 64
  user_vocabulary_size = 2048
  user_embedding_size = 64

  # Split data into training/test data.
  shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
  train = shuffled.take(80_000)
  test = shuffled.skip(80_000).take(20_000)

  # Batch data.
  train_dataset = train.batch(batch_size, drop_remainder=True).cache()
  test_dataset = test.batch(batch_size, drop_remainder=True).cache()

  # Print some values.
  ratings_iter = iter(train_dataset)
  for _ in range(10):
    print(next(ratings_iter))


  # Embedding layer configurations.
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
          name="movie",
          table=movie_table,
          input_shape=(batch_size, None),
          output_shape=(batch_size, movie_embedding_size),
      ),
      "user_id": keras_rs.layers.FeatureConfig(
          name="user",
          table=user_table,
          input_shape=(batch_size, None),
          output_shape=(batch_size, user_embedding_size),
      ),
  }

  # Set up distribution.
  keras.distribution.set_distribution(keras.distribution.DataParallel())

  # Construct model.
  model = EmbeddingModel(feature_config)
  model.compile(
      loss=keras.losses.MeanSquaredError(),
      metrics=[keras.metrics.RootMeanSquaredError()],
      optimizer="adagrad",
  )

  # Set up preprocessing.
  def train_dataset_generator():
    for inputs, labels in iter(train_dataset):
      yield model.embedding_layer.preprocess(inputs, training=True), labels

  def test_dataset_generator():
    for inputs, labels in iter(test_dataset):
      yield model.embedding_layer.preprocess(inputs, training=True), labels

  # Train model.
  model.fit(
      train_dataset_generator(),
      validation_data=test_dataset_generator(),
      validation_freq=5,
      epochs=50,
  )


if __name__ == "__main__":
  app.run(main)
