import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

from VAE import VAE
from Sampling import Sampling


class CAN(keras.Model):
    """
    CAN - Concept Assocation Network 
    Versus a SCAN structure (Symbol Concept Assocation Network), this
    class interprets two continuous sterams of data instead of 
    taking in a continuous stream of data and a classification label.
    """

    def __init__(
        self, vae, can_data, vae_data, latent_dim, epochs, batch_size, **kwargs
    ):
        super(CAN, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_bimodal_loss_tracker = keras.metrics.Mean(name="kl_bimodal_loss")

        self.trained_encoder = vae.encoder
        self.trained_decoder = vae.decoder

        self.can_data_sum = np.array(
            [np.sum(can_data[i, :, :, :]) for i in range(can_data.shape[0])]
        )
        self.vae_data = vae_data
        self.epochs = epochs
        self.batch_size = batch_size

        encoder_inputs = keras.Input(shape=(2, 140, 1))
        x = layers.Conv2D(8, 2, activation="relu", strides=2, padding="same")(
            encoder_inputs
        )
        x = layers.Conv2D(8, 2, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="san_encoder"
        )
        self.encoder = encoder

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(1 * 35 * 8, activation="relu")(latent_inputs)
        x = layers.Reshape((1, 35, 8))(x)
        x = layers.Conv2DTranspose(
            8, 2, activation="relu", strides=(1, 2), padding="same"
        )(x)
        x = layers.Conv2DTranspose(8, 2, activation="relu", strides=2, padding="same")(
            x
        )
        decoder_outputs = layers.Conv2DTranspose(
            1, 3, activation="sigmoid", padding="same"
        )(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="san_decoder")
        self.decoder = decoder

    @property
    def metrics(self):
        """
        Returns:
            List of total loss, reconstruction loss, KL loss between
            the distribution of latent variables to actual, and the KL
            loss between another data modalities distribution and the
            distribution of its own.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_bimodal_loss_tracker,
        ]

    def train_step(self, data):
        """
        Assume Bernoulli distribution for likelihood on each pixel (hence use binary
        cross entropy for reconstruction error). If assuming Gaussian error, 
        for the reconstruction metric use MSE in place of this (may end up 
        focusing on a few pixels that are very wrong though). 

        KL Divergence is measured as the 

        Input: 
            Takes in pupil data of size ( , 2, 140).

        Returns:
            Returns training loss of the model trained on the data.
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.MSE(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            data_np = data.numpy()
            data_sum = np.array(
                [np.sum(data_np[i, :, :, :]) for i in range(data_np.shape[0])]
            )
            idx = np.nonzero(np.in1d(self.can_data_sum, data_sum))
            vae_batch_data = np.take(self.vae_data, idx, axis=0)
            vae_batch_data = vae_batch_data.reshape(vae_batch_data.shape[1:])
            z_vae_mean, z_vae_log_var, z_vae = self.trained_encoder(vae_batch_data)
            vae_data_dist = tfp.distributions.Normal(z_vae_mean, tf.exp(z_vae_log_var))
            data_dist = tfp.distributions.Normal(z_mean, tf.exp(z_log_var))
            kl_bimodal_loss = tfp.distributions.kl_divergence(vae_data_dist, data_dist)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss) + kl_bimodal_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_bimodal_loss_tracker.update_state(kl_bimodal_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_bimodal_loss": self.kl_bimodal_loss_tracker.result(),
        }
