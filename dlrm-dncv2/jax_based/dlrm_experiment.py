"""DLRM experiment."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Generic, Literal, TypeVar

from etils import epy
import fiddle as fdl
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import recml
from recml.layers.linen import sparsecore
import tensorflow as tf

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


# Define feature structures
@dataclasses.dataclass
class Feature:
  name: str


FeatureT = TypeVar('FeatureT', bound=Feature)


@dataclasses.dataclass
class DenseFeature(Feature):
  """Dense feature."""


@dataclasses.dataclass
class SparseFeature(Feature):
  """Sparse feature."""

  vocab_size: int
  embedding_dim: int
  max_sequence_length: int
  combiner: Literal['mean', 'sum', 'sqrtn'] = 'mean'


@dataclasses.dataclass
class FeatureSet(Generic[FeatureT]):
  """A collection of features."""

  features: Sequence[FeatureT]

  def __post_init__(self):
    feature_names = [f.name for f in self.features]
    if len(feature_names) != len(set(feature_names)):
      raise ValueError(
          f'Feature names must be unique. Got names: {feature_names}.'
      )

  def dense_features(self) -> FeatureSet[DenseFeature]:
    return FeatureSet[DenseFeature](
        [f for f in self if isinstance(f, DenseFeature)]
    )

  def sparse_features(self) -> FeatureSet[SparseFeature]:
    return FeatureSet[SparseFeature](
        [f for f in self if isinstance(f, SparseFeature)]
    )

  def __iter__(self) -> Iterator[FeatureT]:
    return iter(self.features)


# Define the DLRM model using Flax
class DLRMModel(nn.Module):
  """DLRM DCN v2 model."""

  features: FeatureSet
  embedding_optimizer: sparsecore.OptimizerSpec
  bottom_mlp_dims: Sequence[int]
  top_mlp_dims: Sequence[int]
  dcn_layers: int
  dcn_inner_dim: int
  _sparsecore_config: sparsecore.SparsecoreConfig | None = None

  @property
  def sparsecore_config(self) -> sparsecore.SparsecoreConfig:
    if self._sparsecore_config is not None:
      return self._sparsecore_config

    sparsecore_config = sparsecore.SparsecoreConfig(
        specs={
            f.name: sparsecore.EmbeddingSpec(
                input_dim=f.vocab_size,
                embedding_dim=f.embedding_dim,
                max_sequence_length=f.max_sequence_length,
                combiner=f.combiner,
            )
            for f in self.features.sparse_features()
        },
        optimizer=self.embedding_optimizer,
    )
    object.__setattr__(self, '_sparsecore_config', sparsecore_config)
    return sparsecore_config

  def bottom_mlp(self, inputs: Mapping[str, jt.Array]) -> jt.Array:
    x = jnp.concatenate(
        [inputs[f.name] for f in self.features.dense_features()], axis=-1
    )
    for dim in self.bottom_mlp_dims:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)
    return x

  def top_mlp(self, x: jt.Array) -> jt.Array:
    for dim in self.top_mlp_dims[:-1]:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)
    x = nn.Dense(self.top_mlp_dims[-1])(x)
    return x

  def dcn(self, x0: jt.Array) -> jt.Array:
    xl = x0
    input_dim = x0.shape[-1]
    for i in range(self.dcn_layers):
      u_kernel = self.param(
          f'u_kernel_{i}',
          nn.initializers.xavier_normal(),
          (input_dim, self.dcn_inner_dim),
      )
      v_kernel = self.param(
          f'v_kernel_{i}',
          nn.initializers.xavier_normal(),
          (self.dcn_inner_dim, input_dim),
      )
      bias = self.param(f'bias_{i}', nn.initializers.zeros, (input_dim,))
      u = jnp.matmul(xl, u_kernel)
      v = jnp.matmul(u, v_kernel)
      v += bias
      xl = x0 * v + xl
    return xl

  @nn.compact
  def __call__(
      self, inputs: Mapping[str, jt.Array], training: bool = False
  ) -> jt.Array:
    dense_embeddings = self.bottom_mlp(inputs)
    sparse_embeddings = sparsecore.SparsecoreEmbed(
        self.sparsecore_config, name='sparsecore_embed'
    )(inputs)
    sparse_embeddings = jax.tree.flatten(sparse_embeddings)[0]
    concatenated_embeddings = jnp.concatenate(
        (dense_embeddings, *sparse_embeddings), axis=-1
    )
    interaction_outputs = self.dcn(concatenated_embeddings)
    predictions = self.top_mlp(interaction_outputs)
    predictions = jnp.reshape(predictions, (-1,))
    return predictions


# Factory to load real data from TFRecords
class TFRecordCriteoFactory(recml.Factory[tf.data.Dataset]):
  """Loads Criteo data from TFRecord files."""

  features: FeatureSet
  input_path: str
  global_batch_size: int
  is_training: bool

  def make(self) -> tf.data.Dataset:
    dense_features = self.features.dense_features()
    sparse_features = self.features.sparse_features()
    num_dense = len(dense_features.features)
    num_sparse = len(sparse_features.features)

    # Define the parsing spec based on the feature set
    feature_spec = {'label': tf.io.FixedLenFeature([1], dtype=tf.int64)}
    for i in range(num_dense):
      feature_spec[f'dense-feature-{i+1}'] = tf.io.FixedLenFeature([1], dtype=tf.float32)
    for i in range(num_sparse):
      feature_spec[f'sparse-feature-{i+num_dense+1}'] = tf.io.VarLenFeature(dtype=tf.int64)

    def _parse_fn(features_proto):
      parsed_features = tf.io.parse_example(features_proto, feature_spec)
      label = tf.cast(parsed_features.pop('label'), tf.float32)

      # Reconstruct the input dictionary for the model
      model_inputs = {}
      dense_list = []
      for i, f in enumerate(dense_features):
        dense_list.append(parsed_features[f'dense-feature-{i+1}'])
      # Use the name of the first dense feature for the concatenated tensor
      model_inputs[dense_features.features[0].name] = tf.concat(dense_list, axis=1)

      # Handle dense features for the model (assuming one concatenated feature)
      # This is a simplification; you might need to adjust based on model needs.
      dense_concat = tf.concat([parsed_features[f'dense-feature-{i+1}'] for i in range(num_dense)], axis=1)
      for i, f in enumerate(dense_features):
          model_inputs[f.name] = tf.expand_dims(dense_concat[:, i], axis=1)

      for i, f in enumerate(sparse_features):
        sparse_tensor = parsed_features[f'sparse-feature-{i+num_dense+1}']
        dense_tensor = tf.sparse.to_dense(sparse_tensor)
        # Pad to max_sequence_length
        padding = [[0, 0], [0, f.max_sequence_length - tf.shape(dense_tensor)[1]]]
        model_inputs[f.name] = tf.pad(dense_tensor, padding)

      return model_inputs, label

    dataset = tf.data.Dataset.list_files(self.input_path, shuffle=self.is_training)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not self.is_training,
    )
    dataset = dataset.batch(self.global_batch_size, drop_remainder=self.is_training)
    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)

# Main task definition
@dataclasses.dataclass
class PredictionTask(recml.JaxTask):
  """Prediction task."""
  train_data: recml.Factory[tf.data.Dataset]
  eval_data: recml.Factory[tf.data.Dataset]
  model: DLRMModel
  optimizer: recml.Factory[optax.GradientTransformation]

  def create_datasets(self) -> tuple[recml.data.Iterator, recml.data.Iterator]:
    global_batch_size = self.train_data.global_batch_size
    train_iter = recml.data.TFDatasetIterator(
        dataset=self.train_data.make(),
        postprocessor=sparsecore.SparsecorePreprocessor(
            self.model.sparsecore_config, global_batch_size
        ),
    )
    eval_iter = recml.data.TFDatasetIterator(
        dataset=self.eval_data.make(),
        postprocessor=sparsecore.SparsecorePreprocessor(
            self.model.sparsecore_config, global_batch_size
        ),
    )
    return train_iter, eval_iter

  def create_state(self, batch: jt.PyTree, rng: jt.Array) -> recml.JaxState:
    inputs, _ = batch
    params = self.model.init(rng, inputs)
    optimizer = self.optimizer.make()
    return recml.JaxState.create(params=params, tx=optimizer)

  def train_step(
      self, batch: jt.PyTree, state: recml.JaxState, rng: jt.Array
  ) -> tuple[recml.JaxState, Mapping[str, recml.Metric]]:
    inputs, label = batch
    def _loss_fn(params: jt.PyTree) -> tuple[jt.Scalar, jt.Array]:
      logits = self.model.apply(params, inputs, training=True)
      loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, label), axis=0)
      return loss, logits
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.update(grads=grads)

    # Calculate the number of examples in the global batch
    num_examples = self.train_data.global_batch_size

    metrics = {
        'loss': recml.metrics.scalar(loss),
        'accuracy': recml.metrics.binary_accuracy(label, logits, threshold=0.0),
        'throughput': recml.metrics.Throughput(num_examples),
    }
    return state, metrics

  def eval_step(
      self, batch: jt.PyTree, state: recml.JaxState
  ) -> Mapping[str, recml.Metric]:
    inputs, label = batch
    logits = self.model.apply(state.params, inputs, training=False)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, label), axis=0)

    # Calculate the number of examples in the global batch
    num_examples = self.eval_data.global_batch_size

    metrics = {
        'loss': recml.metrics.mean(loss),
        'accuracy': recml.metrics.binary_accuracy(label, logits, threshold=0.0),
        'throughput': recml.metrics.Throughput(num_examples),
    }
    return metrics


# Fiddle configuration for the feature set
def features() -> fdl.Config[FeatureSet]:
  """Creates a feature collection for the DLRM model."""
  # Vocab sizes from the Criteo Terabyte dataset
  vocab_sizes = [
      40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
      3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
      40000000, 40000000, 590152, 12973, 108, 36
  ]
  # Corresponding max sequence lengths for each sparse feature
  multi_hot_sizes = [
      3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100,
      27, 10, 3, 1, 1
  ]

  # Pad vocab sizes to be divisible by 1024 for TPU performance
  padded_vocab_sizes = [((v + 1023) // 1024) * 1024 for v in vocab_sizes]

  embedding_dim = 128 # Using a uniform embedding dimension

  return fdl.Config(
      FeatureSet,
      features=[
          fdl.Config(DenseFeature, name=f'dense-feature-{i}') for i in range(13)
      ]
      + [
          fdl.Config(
              SparseFeature,
              name=f'sparse-feature-{i}',
              vocab_size=padded_vocab_sizes[i],
              embedding_dim=embedding_dim,
              max_sequence_length=multi_hot_sizes[i],
          )
          for i in range(len(padded_vocab_sizes))
      ],
  )


# Fiddle configuration for the entire experiment
def experiment() -> fdl.Config[recml.Experiment]:
  """DLRM experiment."""
  feature_set = features()
  task = fdl.Config(
      PredictionTask,
      train_data=fdl.Config(
          TFRecordCriteoFactory,
          features=feature_set,
          is_training=True
          # input_path and global_batch_size will be set from flags
      ),
      eval_data=fdl.Config(
          TFRecordCriteoFactory,
          features=feature_set,
          is_training=False
          # input_path and global_batch_size will be set from flags
      ),
      model=fdl.Config(
          DLRMModel,
          features=feature_set,
          embedding_optimizer=fdl.Config(
              embedding_spec.AdagradOptimizerSpec,
              learning_rate=0.01,
          ),
          bottom_mlp_dims=[512, 256, 128],
          top_mlp_dims=[1024, 1024, 512, 256, 1],
          dcn_layers=3,
          dcn_inner_dim=512,
      ),
      optimizer=fdl.Config(
          recml.AdagradFactory,
          learning_rate=0.01,
          freeze_mask=rf'.*{sparsecore.EMBEDDING_PARAM_NAME}.*',
      ),
  )
  trainer = fdl.Config(
      recml.JaxTrainer,
      partitioner=fdl.Config(recml.DataParallelPartitioner),
      checkpointer=fdl.Config(recml.Checkpointer), # Will be configured from flags
      train_steps=10000,
      steps_per_eval=1000,
      steps_per_loop=1000,
  )
  return fdl.Config(recml.Experiment, task=task, trainer=trainer)

