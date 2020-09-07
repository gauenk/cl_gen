import numpy as np

def add_noise(noise,pic):
    noisy_pic = pic + noise
    return noisy_pic

def np_divide(np_array_a,np_array_b):
    not_np = False
    if type(np_array_a) is not np.ndarray:
        not_np = True
        if type(np_array_a) is not list:
            np_array_a = [np_array_a]
        np_array_a = np.array(np_array_a)
    ma_a = np.ma.masked_array(np_array_a)
    div = (ma_a / np_array_b).filled(0)
    if not_np:
        div = div[0]
    return div

def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)

