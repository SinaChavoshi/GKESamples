"""DLRM DCN v2 model implemented in Flax for JAX."""

from typing import List, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp

# Define a uniform initializer function for weights, as used in the reference.
def uniform_init(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key, shape=shape, dtype=dtype, minval=-bound, maxval=bound
        )
    return init

class DLRMDCNV2(nn.Module):
    """A Flax implementation of the DLRM-DCNv2 model."""
    vocab_sizes: Sequence[int]
    num_dense_features: int = 13
    embedding_dim: int = 128
    bottom_mlp_dims: Sequence[int] = (512, 256, 128)
    dcn_num_layers: int = 3
    dcn_low_rank_dim: int = 512
    top_mlp_dims: Sequence[int] = (1024, 1024, 512, 256)

    def setup(self):
        """Initializes the layers of the model."""
        # --- Embedding Tables ---
        # We create a list of embedding tables, one for each sparse feature.
        self.embedding_tables = [
            nn.Embed(
                num_embeddings=vs,
                features=self.embedding_dim,
                name=f"embed_{i}",
                embedding_init=uniform_init(1.0 / jnp.sqrt(vs)),
            ) for i, vs in enumerate(self.vocab_sizes)
        ]

        # --- Bottom MLP for processing dense features ---
        bottom_mlp_layers = []
        for i, dim in enumerate(self.bottom_mlp_dims):
            bottom_mlp_layers.append(nn.Dense(features=dim, name=f"bottom_mlp_{i}"))
            bottom_mlp_layers.append(nn.relu)
        self.bottom_mlp = nn.Sequential(bottom_mlp_layers)

        # --- DCN V2 Cross Network Layers ---
        # Using low-rank decomposition (U and V matrices) for efficiency.
        self.v_kernels = [
            nn.Dense(features=self.dcn_low_rank_dim, use_bias=False, name=f"dcn_v_{i}")
            for i in range(self.dcn_num_layers)
        ]
        # The output dimension of U must match the concatenated feature dimension.
        # This will be calculated based on the input shape in the __call__ method.
        self.u_kernels = [
            nn.Dense(features=self.num_dense_features + len(self.vocab_sizes) * self.embedding_dim, use_bias=False, name=f"dcn_u_{i}")
            for i in range(self.dcn_num_layers)
        ]
        self.dcn_biases = [
             self.param(f"dcn_bias_{i}", nn.initializers.zeros, (self.num_dense_features + len(self.vocab_sizes) * self.embedding_dim,))
             for i in range(self.dcn_num_layers)
        ]

        # --- Top MLP ---
        top_mlp_layers = []
        for i, dim in enumerate(self.top_mlp_dims):
            top_mlp_layers.append(nn.Dense(features=dim, name=f"top_mlp_{i}"))
            top_mlp_layers.append(nn.relu)
        top_mlp_layers.append(nn.Dense(features=1, name="top_mlp_final"))
        self.top_mlp = nn.Sequential(top_mlp_layers)

    def __call__(self, dense_features, sparse_features_dict):
        """Defines the forward pass of the model."""
        # Process dense features through the bottom MLP.
        dense_x = self.bottom_mlp(dense_features)

        # Process sparse features: embedding lookup and mean reduction.
        sparse_embeds = []
        for i, emb_table in enumerate(self.embedding_tables):
            # The model receives sparse features as a dictionary.
            sparse_ids = sparse_features_dict[str(i)]
            embedding_vector = emb_table(sparse_ids)
            # Average embeddings for multi-hot features.
            sparse_embeds.append(jnp.mean(embedding_vector, axis=1))

        # Concatenate dense and sparse features to form the DCN input.
        x0 = jnp.concatenate([dense_x] + sparse_embeds, axis=1)

        # DCN V2 Cross Network forward pass.
        xl = x0
        for i in range(self.dcn_num_layers):
            v_out = self.v_kernels[i](xl)
            u_out = self.u_kernels[i](v_out)
            xl = x0 * (u_out + self.dcn_biases[i]) + xl

        # Concatenate bottom MLP output with DCN output for the top MLP.
        top_mlp_input = jnp.concatenate([dense_x, xl], axis=1)
        
        # Final prediction logits.
        logits = self.top_mlp(top_mlp_input)
        
        return jnp.squeeze(logits)

