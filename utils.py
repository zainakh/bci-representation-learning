import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import time


def read_single_trial_datasets(data_root, normalization="z"):
    """
    Returns:
        Data queues of single trial EEG and pupil data.
    """
    eeg_train, pupil_train = [], []
    eeg_test, pupil_test = [], []

    eeg_root_dir = data_root + "/EEG_Data/"
    pupil_root_dir = data_root + "/Pupil_Data/"

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
                if normalization == "z":
                    eeg_images[:, :, trial] = z_score_normalize(eeg_images[:, :, trial])
                    pupil_images[:, :, trial] = z_score_normalize(
                        pupil_images[:, :, trial]
                    )
                elif normalization == "mm":
                    eeg_images[:, :, trial] = min_max_normalize(eeg_images[:, :, trial])
                    pupil_images[:, :, trial] = min_max_normalize(
                        pupil_images[:, :, trial]
                    )

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

def read_dataset_by_condition(data_root, normalization="z"):
    """
    Returns:
        Dataset split by condition of free vs eye.
    """
    eeg_free, eeg_eye = [], []
    pupil_free, pupil_eye = [], []

    eeg_root_dir = data_root + "/EEG_Data/"
    pupil_root_dir = data_root + "/Pupil_Data/"

    subjects = [i for i in range(8, 17)]
    conditions = ["free", "eye"]
    for sub in subjects:
        for condition in conditions:
            eeg_path = eeg_root_dir + "EEG_s" + str(sub) + "_" + condition
            eeg_dict = sio.loadmat(eeg_path + "_targets.mat")
            eeg_targ = eeg_dict["eeg_data"]
            eeg_dict = sio.loadmat(eeg_path + "_distractors.mat")
            eeg_dist = eeg_dict["eeg_data"]

            pupil_path = pupil_root_dir + "pupil_s" + str(sub) + "_" + condition
            pupil_dict = sio.loadmat(pupil_path + "_targets.mat")
            pupil_targ = pupil_dict["pupil_data"]
            pupil_dict = sio.loadmat(pupil_path + "_distractors.mat")
            pupil_dist = pupil_dict["pupil_data"]

            eeg_targ = np.resize(eeg_targ, eeg_dist.shape)
            pupil_targ = np.resize(pupil_targ, pupil_dist.shape)

            eeg_images = np.concatenate([eeg_targ, eeg_dist], axis=1)
            pupil_images = np.concatenate([pupil_targ, pupil_dist], axis=1)

            trials = eeg_images.shape[-1]
            pupil_images = pupil_images[:, :, 0:trials]
            pupil_images = pupil_images[(0, 5), :, :]

            for trial in range(trials):
                if normalization == "z":
                    eeg_images[:, :, trial] = z_score_normalize(eeg_images[:, :, trial])
                    pupil_images[:, :, trial] = z_score_normalize(
                        pupil_images[:, :, trial]
                    )
                elif normalization == "mm":
                    eeg_images[:, :, trial] = min_max_normalize(eeg_images[:, :, trial])
                    pupil_images[:, :, trial] = min_max_normalize(
                        pupil_images[:, :, trial]
                    )  
            if condition == "free":
                eeg_free.append(eeg_images)
                pupil_free.append(pupil_images)
            elif condition == "eye":
                eeg_eye.append(eeg_images)
                pupil_eye.append(pupil_images)
                
    for i in range(len(eeg_free)):
        eeg_free[i] = np.rollaxis(eeg_free[i], 2, 0)
        eeg_eye[i] = np.rollaxis(eeg_eye[i], 2, 0)
        pupil_free[i] = np.rollaxis(pupil_free[i], 2, 0)
        pupil_eye[i] = np.rollaxis(pupil_eye[i], 2, 0)
        
        eeg_free[i] = (eeg_free[i])[..., np.newaxis]
        eeg_eye[i] = (eeg_eye[i])[..., np.newaxis]
        pupil_free[i] = (pupil_free[i])[..., np.newaxis]
        pupil_eye[i] = (pupil_eye[i])[..., np.newaxis]

    return (eeg_free, eeg_eye, pupil_free, pupil_eye)


def remove_noisy_trials(eeg_train, eeg_test):
    noisy_train = np.load('noisy_train.npy')
    noisy_test = np.load('noisy_test.npy')
    
    eeg_train = np.delete(eeg_train, noisy_train, axis=0)
    eeg_test = np.delete(eeg_test, noisy_test, axis=0)
    
    return (eeg_train, eeg_test)


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


def get_filename(prefix, name):
    """
    Returns:
        Datetime in form of string with string prefix and suffix.
    """
    timestr = time.strftime("-%Y%m%d-%H%M%S")
    filename = prefix + name + timestr
    return filename


def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def z_score_normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)
