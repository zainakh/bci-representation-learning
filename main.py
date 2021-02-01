import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from utils import read_data_sets, get_filename
from Sampling import Sampling
from VAE import VAE
from CAN import CAN


def main():
    # Set parameters
    vae_epoch = 10
    vae_batch_size = 128
    latent_dim = 10
    beta_eeg = 1.0
    test_split = 0.2

    can_epoch = 10
    can_batch_size = 16

    # Read data sets
    erp, pupil = read_data_sets()
    erp_train, erp_test, pupil_train, pupil_test = train_test_split(
        erp, pupil, test_size=test_split
    )

    # Train VAE
    vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(erp_train, epochs=vae_epoch, batch_size=vae_batch_size)

    # VAE predictions
    encoded_data = vae.encoder.predict(erp_test)
    decoded_data = vae.decoder.predict(encoded_data)
    fn = get_filename("data/", "test-erp")
    # np.save(fn, decoded_data)

    encoded_data = vae.encoder.predict(erp)
    decoded_data = vae.decoder.predict(encoded_data)
    fn = get_filename("data/", "all-erp")
    # np.save(fn, decoded_data)

    # Train CAN
    can = CAN(vae=vae, vae_data=erp_train, latent_dim=latent_dim)
    can.compile(optimizer=keras.optimizers.Adam())
    can.fit(pupil_train, epochs=can_epoch, batch_size=can_batch_size)

    # Save models
    pass


if __name__ == "__main__":
    main()
