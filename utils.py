import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import time


def read_erp_datasets():
    """
    Returns:
        Data queues of erp and pupil.
    """
    erp, pupil = [], []

    erp_root_dir = "/home/zainkhan/Desktop/NiDyN/ERP_Images/"
    pupil_root_dir = "/home/zainkhan/Desktop/NiDyN/Pupil_Images/"

    subjects = [i for i in range(8, 17)]
    conditions = ["free", "eye"]

    for sub in subjects:
        for condition in conditions:
            erp_path = (
                erp_root_dir + "ERP_Image_s" + str(sub) + "_" + condition + ".mat"
            )
            erp_dict = sio.loadmat(erp_path)
            erp.append(erp_dict["erp_cat"])

            pupil_path = (
                pupil_root_dir + "Pupil_Image_s" + str(sub) + "_" + condition + ".mat"
            )
            pupil_dict = sio.loadmat(pupil_path)
            pupil.append(pupil_dict["pupil_cat"])

    erp = np.array(erp)
    erp = erp[..., np.newaxis]
    pupil = np.array(pupil)
    pupil = pupil[..., np.newaxis]

    return (erp, pupil)


def read_single_trial_datasets():
    """
    Returns:
        Data queues of single trial EEG and pupil data.
    """
    eeg_train, pupil_train = [], []
    eeg_test, pupil_test = [], []

    eeg_root_dir = "/home/zainkhan/Desktop/EEG_Data/"
    pupil_root_dir = "/home/zainkhan/Desktop/Pupil_Data/"

    subjects = [i for i in range(8, 17)]
    conditions = ["free", "eye"]

    for sub in subjects:
        for condition in conditions:
            # Load EEG targets and distractors
            eeg_path = eeg_root_dir + "EEG_s" + str(sub) + "_" + condition
            eeg_dict = sio.loadmat(eeg_path + "_targets.mat")
            eeg_targ = eeg_dict["eeg_data"]

            eeg_dict = sio.loadmat(eeg_path + "_distractors.mat")
            eeg_dist = eeg_dict["eeg_data"]

            # Load pupil targets and distractors
            pupil_path = pupil_root_dir + "pupil_s" + str(sub) + "_" + condition
            pupil_dict = sio.loadmat(pupil_path + "_targets.mat")
            pupil_targ = pupil_dict["pupil_data"]

            pupil_dict = sio.loadmat(pupil_path + "_distractors.mat")
            pupil_dist = pupil_dict["pupil_data"]

            # Oversample targets
            eeg_targ = np.resize(eeg_targ, eeg_dist.shape)
            pupil_targ = np.resize(pupil_targ, pupil_dist.shape)

            # Concatenate targets and distractors
            eeg_images = np.concatenate([eeg_targ, eeg_dist], axis=1)
            pupil_images = np.concatenate([pupil_targ, pupil_dist], axis=1)

            # Remove additional trails that pupil may have that EEG does not
            trials = eeg_images.shape[-1]
            pupil_images = pupil_images[:, :, 0:trials]

            # Pupil only concerned with left and right eye pupil size streams
            pupil_images = pupil_images[(0, 5), :, :]

            # Min max normalize
            for trial in range(trials):
                eeg_images[:, :, trial] = min_max_normalize(eeg_images[:, :, trial])
                pupil_images[:, :, trial] = min_max_normalize(pupil_images[:, :, trial])

            if (not len(eeg_train)) and (not len(pupil_train)):
                eeg_train = eeg_images
                pupil_train = pupil_images
            else:
                if sub > 14:
                    if (not len(eeg_test)) and (not len(pupil_test)): 
                        eeg_test = eeg_images
                        pupil_test = pupil_images
                    else:
                        eeg_test = np.dstack([eeg_test, eeg_images])
                        pupil_test = np.dstack([pupil_test, pupil_images])
                else:
                    eeg_train = np.dstack([eeg_train, eeg_images])
                    pupil_train = np.dstack([pupil_train, pupil_images])

    eeg_train = np.rollaxis(eeg_train, 2, 0)
    pupil_train = np.rollaxis(pupil_train, 2, 0)
    eeg_test = np.rollaxis(eeg_test, 2, 0)
    pupil_test = np.rollaxis(pupil_test, 2, 0)

    eeg_train = eeg_train[..., np.newaxis]
    pupil_train = pupil_train[..., np.newaxis]
    eeg_test = eeg_test[..., np.newaxis]
    pupil_test = pupil_test[..., np.newaxis]

    return (eeg_train, eeg_test, pupil_train, pupil_test)


def get_filename(prefix, name):
    """
    Returns:
        Datetime in form of string with string prefix and suffix.
    """
    timestr = time.strftime("-%Y_%m_%d-%H_%M")
    filename = prefix + name + timestr
    return filename


def display_errors(image, reconstruction):
    """
    Prints RMSE error value based on the reconstructed image.
    """
    from sklearn.metrics import mean_squared_error

    return np.sqrt(mean_squared_error(image, reconstruction))


def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
