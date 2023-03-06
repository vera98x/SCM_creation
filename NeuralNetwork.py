from causallearn.graph.GeneralGraph import GeneralGraph
from csv_to_df import retrieveDataframe
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import RMSprop
import scipy.special as sps

def negative_log_like(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def gamma_loss(y_true, y_pred):
    print(y_true)
    gamma_distr = tfp.distributions.Gamma(concentration=2, rate=2)
    return -tf.reduce_mean(gamma_distr.log_prob(y_true))

def mse (y_true, y_pred):
    return tf.square (y_true - y_pred)

def firstNN(x_raw, y_raw):
    input_dim = len(x_raw[0])
    x = tf.constant(x_raw)
    y = tf.constant(y_raw)
    print(x.shape, y.shape)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                200, input_dim=input_dim, activation="sigmoid", kernel_initializer="random_uniform"
            ),
            tf.keras.layers.Dense(
                1, input_dim = 20, activation="relu", kernel_initializer="random_uniform"
            ),
        ]
    )
    model.compile(optimizer='adam',
                  loss=mse,
                  metrics=['accuracy', 'poisson', tf.keras.metrics.MeanSquaredError()])
    model.fit(x, y, epochs=80)
    print(model.evaluate(x,  y, verbose=2))