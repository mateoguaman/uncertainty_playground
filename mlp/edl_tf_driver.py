import functools
import numpy as np
import matplotlib.pyplot as plt

import evidential_deep_learning as edl
import tensorflow as tf

rc = 1e-4       # regularizing coefficient - 1e-2
lr = 5e-3       # learning rate            - 5e-4
mlp = 64        # mlp size                 - 64
epochs = 2500   # num epochs               - 2500
nf = 3.0        # noise factor             - 3.0
layers = 4      # number of layers         - 2.0
data = 1000     # datapoints for train/test- 1000
run = 2
testing = 'KL'

def main():
    global rc
    global lr
    global mlp
    global epochs
    global nf
    global layers

    # Create some training and testing data
    # x_train, y_train = my_data(-10, 10, 1000)
    # x_test, y_test = my_data(-20, 20, 1000, train=False)

    x_train = np.load("/home/micah/airlab/uncertainty_playground/mlp/data/more_balanced/imbalanced/inputs.npy").reshape(-1, 1)
    y_train = np.load("/home/micah/airlab/uncertainty_playground/mlp/data/more_balanced/imbalanced/labels.npy").reshape(-1, 1)
    x_test = np.load("/home/micah/airlab/uncertainty_playground/mlp/data/more_balanced/ground_truth/inputs.npy").reshape(-1, 1)
    y_test = np.load("/home/micah/airlab/uncertainty_playground/mlp/data/more_balanced/ground_truth/labels.npy").reshape(-1, 1)

    # Define our model with an evidential output
    if layers == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp, activation="relu"),
            tf.keras.layers.Dense(mlp, activation="relu"),
            edl.tflayers.DenseNormalGamma(1),
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp, activation="relu"),
            tf.keras.layers.Dense(mlp, activation="relu"),
            tf.keras.layers.Dense(mlp, activation="relu"),
            tf.keras.layers.Dense(mlp, activation="relu"),
            edl.tflayers.DenseNormalGamma(1),
        ])

    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return edl.tflosses.EvidentialRegression(true, pred, rc) #1e-2 origin, 1e-12 last

    # Compile and fit the model!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=EvidentialRegressionLoss)
    model.fit(x_train, y_train, batch_size=100, epochs=epochs)

    # Predict and plot using the trained model
    y_pred = model(x_test)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)

    # Done!!


#### Helper functions ####
def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = nf * np.ones_like(x) if train else np.zeros_like(x)
    y = (2*x-4) * np.sin(x) + np.random.normal(0, sigma).astype(np.float32)

    return x, y

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, alpha=0.7, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', alpha=0.7,zorder=2, label="True")
    plt.plot(x_test, mu, alpha=0.7, color='#007cab', zorder=3, label="Pred")
    plt.plot([-20, -20], [-50, 50], 'k--', alpha=0.4, zorder=0)
    plt.plot([+20, +20], [-50, 50], 'k--', alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    plt.gca().set_ylim(-50, 50)
    plt.gca().set_xlim(-20, 20)
    plt.legend(loc="upper left")
    plt.savefig(f'results/{testing}/sine_mlp{mlp}_lr{lr}_rc{rc}_nf{nf}_epo{epochs}_layers{layers}_run{run}_unbalanced.png')
    plt.show()
if __name__ == "__main__":
    main()