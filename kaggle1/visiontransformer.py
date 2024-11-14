import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from math import log2
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model


def mlp(x, cf):
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(num_heads=cf["num_heads"], key_dim=cf["hidden_dim"])(x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, cf)
    x = L.Add()([x, skip_2])

    return x


def build_vision_transformer(cf):
    """Inputs"""
    input_shape = (
        cf["num_patches"],
        cf["patch_size"] * cf["patch_size"] * cf["num_channels"],
    )
    inputs = L.Input(input_shape)  ## (None, 256, 3072)

    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)  ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(
        positions
    )  ## (256, 768)
    x = patch_embed + pos_embed  ## (None, 256, 768)

    """ Transformer Encoder """
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    """ Output """
    # Global average pooling
    x = L.GlobalAveragePooling1D()(x)

    # Dense layer for classification
    outputs = L.Dense(cf["num_classes"], activation="softmax")(x)

    return Model(inputs, outputs, name="VisionTransformer_MultiClass")


if __name__ == "__main__":
    config = {}
    config["image_size"] = 512
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["patch_size"] = 16
    config["num_patches"] = (config["image_size"] // config["patch_size"]) ** 2
    config["num_channels"] = 3
    config["num_classes"] = 440  # New parameter for multiclass classification

    model = build_vision_transformer(config)
    model.summary()
