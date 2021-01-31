import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import read_data_sets
from Sampling import Sampling
from VAE import VAE

def main():
    epoch = 1000
    batch_size = 128
    latent_dim = 10
    beta_eeg = 1.
    
    # Train
    erp, _ = read_data_sets()
    erp = np.array(erp)
    erp = erp[..., np.newaxis]
    vae = VAE(beta=beta_eeg, latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(erp, epochs=epoch, batch_size=batch_size)

    encoded_data = vae.encoder.predict(erp)
    decoded_data = vae.decoder.predict(encoded_data)
    np.save('data/test_rec', decoded_data)

    vae.save('vae_model')

if __name__ == '__main__':
    main()