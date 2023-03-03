from causallearn.graph.GeneralGraph import GeneralGraph
from Load_transform_df import retrieveDataframe
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


def firstNN(x, y_raw):
    input_dim = len(x)
    y = tf.constant(y_raw)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                20, input_dim=input_dim, activation="sigmoid", kernel_initializer="random_uniform"
            ),
            tf.keras.layers.Dense(
                1, imput_dim = 20, activation="relu", kernel_initializer="random_uniform"
            ),
        ]
    )
    model.compile(optimizer='adam',
                  loss=gamma_loss,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=5)
    print(model.evaluate(x,  y, verbose=2))



tfd = tfp.distributions
tfpl = tfp.layers



my_file = open("Data/PrimaryDelays/Mp_8100E.txt", "r")
content = my_file.read()
my_file.close()
c = content.replace('[', '').replace(']', '').replace('\n', '').replace(" ", "")
c.split(",")
y_temp = list(map(lambda x: float((int(x)/60)+0.01), c.split(",")))
y_temp = [i for i in y_temp if i >= 0]
y = tf.constant(y_temp)
#y = np.array(list(map(lambda x: float(int(x)/60), c.split(","))))
nr_samples = len(y)
x = np.linspace(0, 10, nr_samples)[:, np.newaxis]
#x = np.zeros(nr_samples)

shape, scale = 2, 2
y1 = tfd.Gamma(shape, scale).sample(nr_samples)
print(x.shape)
print(x)
print(y.shape)
print(y)
print(y1.shape)
print(y1)

#print(gamma_loss(100, 2).numpy())


count, bins, ignored = plt.hist(y, 50, density=True)
#plt.show()

# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(
#             20, input_dim=1, activation="sigmoid", kernel_initializer="random_uniform"
#         ),
#         tf.keras.layers.Dense(
#             2, imput_dim = 20, activation="relu", kernel_initializer="random_uniform"
#         ),
#         tfp.layers.DistributionLambda(
#             lambda t: tfd.Gamma(
#                 concentration=tf.math.softplus(t[:, 0]) + 1e-9,
#                 rate=tf.math.softplus(t[:, 1]) + 1e-9,
#             ),
#         ),
#     ]
# )

def model_builder(hp):
    hp_concentration = hp.Choice("hp_concentration", values=list(np.arange(0.1, 5, 0.1)))
    hp_rate = hp.Choice("hp_rate", values=list(np.arange(0.1, 5, 0.1)))

    model = tf.keras.Sequential()
    model.add(tfp.layers.DistributionLambda(
                    lambda t: tfd.Gamma(
                        concentration=hp_concentration,
                        rate=hp_rate,
                    ),
                ) )
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(loss=negloglik, optimizer=tf.optimizers.Adamax(learning_rate=1e-4))
    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=50,
                     factor=3,
                     seed = 42,
                     directory=None,
                     project_name=None)

print(tuner.search_space_summary())

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x, y, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is ...., and the optimal learning rate for the optimizer
is {best_hps.get('hp_concentration')}, {best_hps.get('hp_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x, y, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x, y, epochs=best_epoch, validation_split=0.2)

y_model = hypermodel(x)
y_sample = y_model.sample(nr_samples)
y_hat = y_model.mean()
y_sd = y_model.stddev()
y_hat_m2sd = y_hat -2 * y_sd
y_hat_p2sd = y_hat + 2*y_sd

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is ...., and the optimal learning rate for the optimizer
is {best_hps.get('hp_concentration')}, {best_hps.get('hp_rate')}.
""")

print("True μ: ", y.numpy().mean())
print("Estimated μ: ", y_hat.numpy().mean())
print("True σ: ",  np.std(y.numpy()))
print("Estimated σ: ", y_sd.numpy().mean())

plt.hist(y, 50, density=True)
plt.hist(y_sample, 50, density=True)

fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax1.scatter(x, y, alpha=0.4, label='data')
ax1.scatter(x, y_sample, alpha=0.4, color='red', label='model sample')
ax1.legend()
ax2.scatter(x, y, alpha=0.4, label='data')
ax2.plot(x, [y_hat]*nr_samples, color='red', alpha=0.8, label='model $\mu$')
ax2.plot(x, [y_hat_m2sd]*nr_samples, color='green', alpha=0.8, label='model $\mu \pm 2 \sigma$')
ax2.plot(x, [y_hat_p2sd]*nr_samples, color='green', alpha=0.8)
ax2.legend()
plt.show()