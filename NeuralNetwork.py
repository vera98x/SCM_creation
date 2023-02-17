from causallearn.graph.GeneralGraph import GeneralGraph
from Load_transform_df import retrieveDataframe
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import RMSprop
import scipy.special as sps

class NN_sample:
    def __init__(self, label, prev_event = 0, prev_platform = 0, dep1 = 0, dep2 = 0):
        self.label = label
        self.prev_event = prev_event
        self.prev_platform = prev_platform
        self.dep1 = dep1
        self.dep2 = dep2

    def gg_to_nn_input(gg : GeneralGraph):
        nodes = gg.get_nodes()

        for node in nodes:
            pass
        return None

def negative_log_like(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def gamma_loss(y_true, y_pred):
    print(y_true)
    gamma_distr = tfp.distributions.Gamma(concentration=2, rate=2)
    return -tf.reduce_mean(gamma_distr.log_prob(y_true))

tfd = tfp.distributions
tfpl = tfp.layers

x = np.linspace(0, 10, 1000)[:, np.newaxis]
shape, scale = 2, 2
y = tfd.Gamma(shape, scale).sample(1000)
print(x.shape)
print(y.shape)
print(gamma_loss(100, 2).numpy())

count, bins, ignored = plt.hist(y, 50, density=True)
#plt.show()

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            20, input_dim=1, activation="sigmoid", kernel_initializer="random_uniform"
        ),
        tf.keras.layers.Dense(
            2, activation="relu", kernel_initializer="random_uniform"
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Gamma(
                concentration=tf.math.softplus(t[:, 0]) + 1e-9,
                rate=tf.math.softplus(t[:, 1]) + 1e-9,
            ),
        ),
    ]
)


negloglik = lambda y, p_y: -p_y.log_prob(y)

model.compile(loss=negloglik, optimizer=tf.optimizers.Adamax(learning_rate=1e-4))
model.fit(x, y, epochs=500, verbose=False)

y_model = model(x)
y_sample = y_model.sample()
y_hat = y_model.mean()
y_sd = y_model.stddev()
y_hat_m2sd = y_hat -2 * y_sd
y_hat_p2sd = y_hat + 2*y_sd

print("True μ: ", 1)
print("Estimated μ: ", y_hat.numpy().mean())
print("True σ: ", 3/8)
print("Estimated σ: ", y_sd.numpy().mean())

plt.hist(y_model, 50, density=True)
plt.hist(y_sample, 50, density=True)

fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax1.scatter(x, y, alpha=0.4, label='data')
ax1.scatter(x, y_sample, alpha=0.4, color='red', label='model sample')
ax1.legend()
ax2.scatter(x, y, alpha=0.4, label='data')
ax2.plot(x, y_hat, color='red', alpha=0.8, label='model $\mu$')
ax2.plot(x, y_hat_m2sd, color='green', alpha=0.8, label='model $\mu \pm 2 \sigma$')
ax2.plot(x, y_hat_p2sd, color='green', alpha=0.8)
ax2.legend()
plt.show()
