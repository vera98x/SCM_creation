
# -----IMPORTANT FOR REPRODUCABILITY
seed_value = 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)
#--------------------------------------------------------

import tensorflow_probability as tfp

from typing import List
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd


def negative_log_like(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def gamma_loss(y_true, y_pred):
    print(y_true)
    gamma_distr = tfp.distributions.Gamma(concentration=2, rate=2)
    return -tf.reduce_mean(gamma_distr.log_prob(y_true))

def mse (y_true, y_pred):
    return tf.square (y_true - y_pred)
def agv_miss(y_true, y_pred):
    return abs(y_pred - y_true)
def within15sec(y_true, y_pred):
    y_tr = y_true.numpy()
    y_pr = y_pred.numpy()
    diff = abs(y_tr - y_pr)
    c = np.count_nonzero(diff <= 15)
    return c/len(diff)

def firstNN(x_raw : List[List[float]], y_raw : List[float], filename : str):
    input_dim = len(x_raw[0])
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x_raw, y_raw, test_size = 0.20, random_state = 42)
    x_train = tf.constant(x_train_raw)
    y_train = tf.constant(y_train_raw)
    x_test = tf.constant(x_test_raw)
    y_test = tf.constant(y_test_raw)

    print(x_train.shape, y_train.shape)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                300, input_dim=input_dim, activation="tanh", kernel_initializer="random_uniform"
            ),
            tf.keras.layers.Dense(
                1, input_dim = 300, activation="PReLU", kernel_initializer="random_uniform"
            ),
        ]
    )
    model.compile(optimizer='adam',
                  loss=mse,
                  metrics=['accuracy', 'poisson', tf.keras.metrics.MeanSquaredError(), agv_miss])
    model.fit(x_train, y_train, epochs=80)
    print("eval train")
    print(model.evaluate(x_train, y_train, verbose=2))
    print("eval test")
    print(model.evaluate(x_test,  y_test, verbose=2))
    print("_______________")
    prediction = model.predict(x_test)
    prediction_flatten = [item for sublist in prediction for item in sublist]

    y_test_list = y_test.numpy().tolist()
    comparison = list(zip(prediction_flatten, y_test_list))
    print(sum([abs(x - y) for x,y in comparison])/len(prediction_flatten))
    print("within 15 sec: ", sum([abs(x - y) <= 15 for x,y in comparison])/len(prediction_flatten))
    df = pd.DataFrame()
    df["prediction"] = prediction_flatten
    df["actual"] = y_test_list
    df["difference"] = df["prediction"] - df["actual"]
    df = df.round(2)
    filename = filename + "_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + ".csv"
    df.to_csv(filename, index = False, sep = ";")
