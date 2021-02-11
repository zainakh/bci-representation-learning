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
    vae_epoch = 100
    vae_batch_size = 128
    latent_dim = 10
    beta_eeg = 5.0
    can_epoch = 100
    can_batch_size = 16
    train = True

    # Read data sets
    eeg, pupil = utils.read_single_trial_datasets()
    eeg_train, eeg_test, pupil_train, pupil_test = train_test_split(
        eeg, pupil, test_size=0.2, shuffle=False
    )

    if train:
        '''
        # Train VAE
        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(eeg_train, epochs=vae_epoch, batch_size=vae_batch_size)

        # Save VAE
        vae.encoder.save("vae_encoder")
        vae.decoder.save("vae_decoder")
        '''

        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.encoder = keras.models.load_model("vae_encoder")
        vae.decoder = keras.models.load_model("vae_decoder")

        # Train CAN
        can = CAN(vae=vae, vae_data=eeg_train, latent_dim=latent_dim)
        can.compile(optimizer=keras.optimizers.Adam())
        can.fit(pupil_train, epochs=can_epoch, batch_size=can_batch_size)

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
