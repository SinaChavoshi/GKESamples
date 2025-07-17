import dataclasses
import tensorflow as tf

# --- FIX: Define constants at the top level so they can be imported ---
NUM_DENSE_FEATURES = 13
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]
MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10,
    3, 1, 1,
]
NUM_SPARSE_FEATURES = len(VOCAB_SIZES)

@dataclasses.dataclass
class DataConfig:
  """Configuration for data loading parameters."""
  global_batch_size: int
  pre_batch_size: int
  is_training: bool

class CriteoDataLoader:
    """Data loader for the pre-processed Criteo TFRecord dataset."""

    def __init__(self, file_pattern: str, params: DataConfig, num_dense_features: int, vocab_sizes: list, multi_hot_sizes: list):
        self._file_pattern = file_pattern
        self._params = params
        self._num_dense_features = num_dense_features
        self._vocab_sizes = vocab_sizes
        self._multi_hot_sizes = multi_hot_sizes

    def _get_feature_spec(self):
        """Returns the feature specification for parsing TFRecord examples."""
        feature_spec = {
            'label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=None)
        }
        for i in range(1, self._num_dense_features + 1):
            feature_spec[f'dense-feature-{i}'] = tf.io.FixedLenFeature(
                [1], dtype=tf.float32, default_value=None
            )
        for i in range(self._num_dense_features + 1, self._num_dense_features + len(self._vocab_sizes) + 1):
            feature_spec[f'sparse-feature-{i}'] = tf.io.VarLenFeature(dtype=tf.int64)
        return feature_spec

    def _parse_fn(self, features):
        """Parses a single tf.train.Example into the format expected by the model."""
        label = tf.cast(features.pop('label'), tf.float32)

        dense_features_list = []
        for i in range(1, self._num_dense_features + 1):
            dense_features_list.append(features[f'dense-feature-{i}'])
        dense_features = tf.concat(dense_features_list, axis=1)

        sparse_features = {}
        for i in range(self._num_dense_features + 1, self._num_dense_features + len(self._vocab_sizes) + 1):
            model_feature_index = i - (self._num_dense_features + 1)
            sparse_ids = tf.sparse.to_dense(features[f'sparse-feature-{i}'])
            sparse_features[f"{model_feature_index}"] = sparse_ids

        return {
            "dense_features": dense_features,
            "sparse_features": sparse_features,
            "clicked": label,
        }

    def get_iterator(self):
        """Creates and returns a tf.data.Dataset iterator."""
        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._params.is_training)
        if self._params.is_training:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self._params.is_training,
        )
        dataset = dataset.batch(self._params.global_batch_size, drop_remainder=self._params.is_training)
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, self._get_feature_spec()),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self._parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return iter(dataset.prefetch(tf.data.AUTOTUNE))
