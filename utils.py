import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def read_data_sets():
    """
    Returns:
        Data queues of erp and pupil.
    """
    erp, pupil = [], []

    erp_root_dir = '/home/zainkhan/Desktop/NiDyN/ERP_Images/'
    pupil_root_dir = '/home/zainkhan/Desktop/NiDyN/Pupil_Images/'

    subjects = [i for i in range(8, 17)]
    conditions = ['free', 'eye']

    for sub in subjects:
        for condition in conditions:
            erp_path = erp_root_dir + 'ERP_Image_s' + str(sub) + '_' + condition + '.mat'
            erp_dict = sio.loadmat(erp_path)
            erp.append(erp_dict['erp_cat'])
            
            pupil_path = pupil_root_dir + 'Pupil_Image_s' + str(sub) + '_' + condition + '.mat'
            pupil_dict = sio.loadmat(pupil_path)
            pupil.append(pupil_dict['pupil_cat'])

    return(erp, pupil)


def plot_reconstructions():
    from sklearn.metrics import mean_squared_error 

    rec = np.load('data/test_rec.npy')
    rec = rec[0]
    rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
    plt.imshow(rec) 
    plt.show()

    erp, _ = read_data_sets()
    third = erp[2]
    second = erp[1]
    erp = erp[0]
    plt.imshow(erp)
    plt.show()

    erp = np.squeeze(erp)
    rec = np.squeeze(rec)
    second = np.squeeze(second)
    rms = np.sqrt(mean_squared_error(erp, rec))
    print(rms)
    rms = np.sqrt(mean_squared_error(erp, second))
    print(rms)
    rms = np.sqrt(mean_squared_error(erp, third))
    print(rms)

#plot_reconstructions()