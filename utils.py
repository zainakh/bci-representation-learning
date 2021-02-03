import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import time


def read_data_sets():
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
    timestr = time.strftime("-%Y_%m_%d-%H_%M")
    filename = prefix + name + timestr
    return filename


def display_errors(image, reconstruction):
    """
    Prints RMSE error value based on the reconstructed image.
    """
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(image, reconstruction))



