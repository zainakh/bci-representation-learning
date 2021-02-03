import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Sampling import Sampling


class VAE(keras.Model):
    """
    This class stores a variational autoencoder setup meant to learn EEG representations when presented
    in the form of an image. 
    """

    def __init__(self, beta, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.beta = beta
        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        encoder_inputs = keras.Input(shape=(64, 768, 1))
        x = layers.Conv2D(32, 6, activation="relu", strides=2, padding="same")(
            encoder_inputs
        )
        x = layers.Conv2D(32, 6, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder = encoder

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16 * 192 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((16, 192, 32))(x)
        x = layers.Conv2DTranspose(32, 6, activation="relu", strides=2, padding="same")(
            x
        )
        x = layers.Conv2DTranspose(32, 6, activation="relu", strides=2, padding="same")(
            x
        )
        decoder_outputs = layers.Conv2DTranspose(
            1, 3, activation="sigmoid", padding="same"
        )(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder = decoder

    @property
    def metrics(self):
        """
        Returns:
            List of total loss, reconstruction loss, and KL loss.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Assume Bernoulli distribution for lielihood on each pixel (hence use binary
        cross entropy for reconstruction error). If assuming Gaussian, use MSE in place
        of this (may end up focusing on a few pixels that are very wrong though). 

        Input: 
            Takes in EEG data of size ( , 64, 768).

        Returns:
            Returns training loss of the model trained on the data.
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
