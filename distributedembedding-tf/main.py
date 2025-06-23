import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import keras_rs
import tensorflow as tf
import tensorflow_datasets as tfds

# 1. Initialize the TPUStrategy. This is required by keras-rs.
# When running on the GKE nodes, TPUClusterResolver will automatically find the TPU.
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU strategy initialized.")
except ValueError as e:
    print(e)
    print("Not running on TPU, falling back to default strategy.")
    strategy = tf.distribute.get_strategy()

print("Number of replicas:", strategy.num_replicas_in_sync)

PER_REPLICA_BATCH_SIZE = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync



ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

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
    100_000, seed=42, reshuffle_each_iteration=False
)

# 2. Data sharding for multi-worker training is still correct.
# We determine the worker index from the environment variables set by JobSet.
worker_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
num_workers = int(os.environ.get("JOBSET_SIZE", 1))

dataset = shuffled_ratings.shard(num_shards=num_workers, index=worker_index)

# Adjust dataset size per worker
train_size_per_worker = 80_000 // num_workers
test_size_per_worker = 20_000 // num_workers

train_ratings = (
    dataset.take(train_size_per_worker).batch(GLOBAL_BATCH_SIZE, drop_remainder=True).cache()
)
test_ratings = (
    dataset.skip(train_size_per_worker)
    .take(test_size_per_worker)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    .cache()
)


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

# 3. Model creation and compilation MUST be inside the strategy scope.
with strategy.scope():
    model = EmbeddingModel(FEATURE_CONFIGS)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
        optimizer="adagrad",
    )

model.fit(train_ratings, epochs=5, verbose=1 if worker_index == 0 else 0)

# 4. Only have the chief worker (index 0) print the final evaluation.
if worker_index == 0:
    print("Evaluation results:")
    model.evaluate(test_ratings, return_dict=True)
