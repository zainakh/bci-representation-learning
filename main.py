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
    vae_epoch = 10000
    vae_batch_size = 128
    latent_dim = 10
    beta_eeg = 1.0
    test_split = 0.2
    can_epoch = 10000
    can_batch_size = 16
    train = False

    # Read data sets
    erp, pupil = utils.read_data_sets()
    erp_train, erp_test, pupil_train, pupil_test = train_test_split(
        erp, pupil, test_size=test_split, random_state=42
    )

    if train:
        # Train VAE
        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(erp_train, epochs=vae_epoch, batch_size=vae_batch_size)

        # Train CAN
        can = CAN(vae=vae, vae_data=erp_train, latent_dim=latent_dim)
        can.compile(optimizer=keras.optimizers.Adam())
        can.fit(pupil_train, epochs=can_epoch, batch_size=can_batch_size)

        # Save models
        vae.encoder.save("vae_encoder")
        vae.decoder.save("vae_decoder")
        can.encoder.save("can_encoder")
        can.decoder.save("can_decoder")
    else:
        # Load all encoders/decoders
        vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
        vae.encoder = keras.models.load_model("vae_encoder")
        vae.decoder = keras.models.load_model("vae_decoder")

        can = CAN(vae=vae, vae_data=erp_train, latent_dim=latent_dim)
        can.encoder = keras.models.load_model("can_encoder")
        can.decoder = keras.models.load_model("can_decoder")

    # VAE predictions
    encoded_data = vae.encoder.predict(erp_test)
    decoded_data = vae.decoder.predict(encoded_data)
    fn = utils.get_filename("predictions/", "test-erp")
    #np.save(fn, decoded_data)
    utils.plot_erp_reconstructions(erp_test, decoded_data)


if __name__ == "__main__":
    main()
