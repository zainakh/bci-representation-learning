import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import utils
from Sampling import Sampling
from VAE import VAE
from CAN import CAN


def main():
    # Set parameters
    vae_epoch = 1000
    can_epoch = 1000
    batch_size = 64
    latent_dim = 10
    beta_eeg = 5.0
    train = True

    # Read data sets
    eeg_train, eeg_test, pupil_train, pupil_test = utils.read_single_trial_datasets()


    if train:
        # Train VAE
        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(eeg_train, epochs=vae_epoch, batch_size=batch_size)

        # Save VAE
        vae.encoder.save("vae_encoder")
        vae.decoder.save("vae_decoder")

        # Train CAN
        can = CAN(
            vae=vae,
            can_data=pupil_train,
            vae_data=eeg_train,
            latent_dim=latent_dim,
            epochs=can_epoch,
            batch_size=batch_size,
        )
        can.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
        can.fit(pupil_train, epochs=can_epoch, batch_size=batch_size, shuffle=False)

        # Save CAN
        can.encoder.save("can_encoder")
        can.decoder.save("can_decoder")
    else:
        # Load all encoders/decoders
        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.encoder = keras.models.load_model("vae_encoder")
        vae.decoder = keras.models.load_model("vae_decoder")

        can = CAN(vae=vae, vae_data=eeg_train, latent_dim=latent_dim)
        can.encoder = keras.models.load_model("can_encoder")
        can.decoder = keras.models.load_model("can_decoder")

    # VAE predictions
    encoded_data = vae.encoder.predict(eeg_test)
    decoded_data = vae.decoder.predict(encoded_data)
    fn = utils.get_filename("predictions/", "test-eeg")
    # np.save(fn, decoded_data)


if __name__ == "__main__":
    main()
