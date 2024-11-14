import numpy as np
import tensorflow as tf

smooth = 1e-15


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    # Clip prediction values to avoid NaN's
    y_pred = tf.clip_by_value(y_pred, 0, 1)

    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / tf.maximum(
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth, 1e-7
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
