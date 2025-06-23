import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import keras_rs
import tensorflow as tf
import tensorflow_datasets as tfds

# The distribution strategy is now handled automatically by TF_CONFIG!

# The per-replica batch size is what each of the 8 TPU cores will see.
PER_REPLICA_BATCH_SIZE = 256
# The global batch size is the total batch size across all replicas.
# TensorFlow's tf.data.Dataset.batch() will use this.
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * 8 # 8 total TPU chips in a 2x4 topology

# --- The rest of your code remains the same ---

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy())

movies_count = movies.cardinality().numpy()

def preprocess_rating(x):
    return (
        {
            "user_id": tf.strings.to_number(x["user_id"], out_type=tf.int32),
            "movie_id": tf.strings.to_number(x["movie_id"], out_type=tf.int32),
        },
        (x["user_rating"] - 1.0) / 4.0,
    )

shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False)

# When using a tf.distribute strategy, you should shard the dataset.
# Each worker will process a portion of the data.
dataset = shuffled_ratings.shard(num_shards=2, index=int(os.environ.get("JOB_COMPLETION_INDEX", 0)))

train_ratings = (
    dataset.take(40_000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True).cache())
test_ratings = (
    dataset.skip(40_000)
    .take(10_000)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    .cache())

EMBEDDING_DIMENSION = 32

movie_table = keras_rs.layers.TableConfig(
    name="movie_table",
    vocabulary_size=movies_count + 1,
    embedding_dim=EMBEDDING_DIMENSION,
    optimizer="adam",
    placement="sparsecore",
)
user_table = keras_rs.layers.TableConfig(
    name="user_table",
    vocabulary_size=users_count + 1,
    embedding_dim=EMBEDDING_DIMENSION,
    optimizer="adam",
    placement="sparsecore",
)

FEATURE_CONFIGS = {
    "movie_id": keras_rs.layers.FeatureConfig(
        name="movie",
        table=movie_table,
        input_shape=(PER_REPLICA_BATCH_SIZE,),
        output_shape=(PER_REPLICA_BATCH_SIZE, EMBEDDING_DIMENSION),
    ),
    "user_id": keras_rs.layers.FeatureConfig(
        name="user",
        table=user_table,
        input_shape=(PER_REPLICA_BATCH_SIZE,),
        output_shape=(PER_REPLICA_BATCH_SIZE, EMBEDDING_DIMENSION),
    ),
}

class EmbeddingModel(keras.Model):
    def __init__(self, feature_configs):
        super().__init__()
        self.embedding_layer = keras_rs.layers.DistributedEmbedding(
            feature_configs=feature_configs
        )
        self.ratings = keras.Sequential(
            [
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )

    def call(self, features):
        embedding = self.embedding_layer(
            {"user_id": features["user_id"], "movie_id": features["movie_id"]}
        )
        return self.ratings(
            keras.ops.concatenate(
                [embedding["user_id"], embedding["movie_id"]],
                axis=1,
            )
        )

# No need for "with strategy.scope()" if TF_CONFIG is set
model = EmbeddingModel(FEATURE_CONFIGS)
model.compile(
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
    optimizer="adagrad",
)

model.fit(train_ratings, epochs=5)

# Only have the chief worker (index 0) print the final evaluation
if int(os.environ.get("JOB_COMPLETION_INDEX", 0)) == 0:
    print("Evaluation results:")
    model.evaluate(test_ratings, return_dict=True)
