import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="behind_goemotions")
class PositionalEmbedding(layers.Layer):
    """Token embedding + positional embedding with padding-mask propagation."""

    def __init__(self, vocab_size, embed_dim, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.seq_length  = seq_length
        self.token_embeddings    = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.position_embeddings = layers.Embedding(seq_length, embed_dim)
        self.support_masking = True

    def call(self, inputs):
        positions = tf.range(self.seq_length)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def compute_mask(self, inputs, mask=None):
        return self.token_embeddings.compute_mask(inputs, mask)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size,
                       "embed_dim":  self.embed_dim,
                       "seq_length": self.seq_length})
        return config


@tf.keras.utils.register_keras_serializable(package="behind_goemotions")
class EncoderBlock(layers.Layer):
    """Transformer Encoder Block with Pre-LN and residual connections."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.ff_dim       = ff_dim
        self.dropout_rate = dropout_rate

        self.attention  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn        = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1      = layers.LayerNormalization(epsilon=1e-6)
        self.norm2      = layers.LayerNormalization(epsilon=1e-6)
        self.drop_attn  = layers.Dropout(dropout_rate)
        self.drop_ffn   = layers.Dropout(dropout_rate)
        self.support_masking = True

    def call(self, x, training=False):
        attn_out = self.attention(self.norm1(x), self.norm1(x), training=training)
        x = x + self.drop_attn(attn_out, training=training)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.drop_ffn(ffn_out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim":    self.embed_dim,
                       "num_heads":    self.num_heads,
                       "ff_dim":       self.ff_dim,
                       "dropout_rate": self.dropout_rate})
        return config


# ------------------------------------------------------------------------------
# Backward compatibility: also register the classes under the default "Custom"
# package, so that checkpoints saved before the package rename (registered_name
# "Custom>PositionalEmbedding" / "Custom>EncoderBlock") still load correctly.
# ------------------------------------------------------------------------------
tf.keras.utils.register_keras_serializable(package="Custom")(PositionalEmbedding)
tf.keras.utils.register_keras_serializable(package="Custom")(EncoderBlock)