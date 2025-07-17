import tensorflow as tf

# Constants matching the benchmark dataset
NUM_DENSE_FEATURES = 13
NUM_SPARSE_FEATURES = 26

class CriteoDataLoader:
    """Data loader for the pre-processed Criteo TFRecord dataset."""

    def __init__(self, file_pattern: str, global_batch_size: int, is_training: bool):
        self._file_pattern = file_pattern
        self._global_batch_size = global_batch_size
        self._is_training = is_training

    def _get_feature_spec(self):
        """Returns the feature specification for parsing TFRecord examples."""
        feature_spec = {
            'label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=None)
        }
        # Dense features are named dense-feature-1, dense-feature-2, ...
        for i in range(1, NUM_DENSE_FEATURES + 1):
            feature_spec[f'dense-feature-{i}'] = tf.io.FixedLenFeature(
                [1], dtype=tf.float32, default_value=None
            )
        # Sparse features are named sparse-feature-14, sparse-feature-15, ...
        for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
            feature_spec[f'sparse-feature-{i}'] = tf.io.VarLenFeature(dtype=tf.int64)
        return feature_spec

    def _parse_fn(self, features):
        """Parses a single tf.train.Example into the format expected by the model."""
        label = tf.cast(features.pop('label'), tf.float32)

        dense_features_list = []
        for i in range(1, NUM_DENSE_FEATURES + 1):
            dense_features_list.append(features[f'dense-feature-{i}'])
        dense_features = tf.concat(dense_features_list, axis=1)

        sparse_features = {}
        for i in range(NUM_DENSE_FEATURES + 1, NUM_DENSE_FEATURES + NUM_SPARSE_FEATURES + 1):
            # The model expects sparse features to be named '0', '1', '2', etc.
            model_feature_index = i - (NUM_DENSE_FEATURES + 1)
            sparse_ids = tf.sparse.to_dense(features[f'sparse-feature-{i}'])
            sparse_features[f"{model_feature_index}"] = sparse_ids

        return {
            "dense_features": dense_features,
            "sparse_features": sparse_features,
            "clicked": label,
        }

    def get_iterator(self):
        """Creates and returns a tf.data.Dataset iterator."""
        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._is_training)
        if self._is_training:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self._is_training,
        )
        dataset = dataset.batch(self._global_batch_size, drop_remainder=self._is_training)
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, self._get_feature_spec()),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self._parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return iter(dataset.prefetch(tf.data.AUTOTUNE))
