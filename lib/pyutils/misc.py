
# python imports
import numpy as np
import pickle,sys,os,yaml,io
from easydict import EasyDict as edict

# this is the only allowed project import in this file.
import settings

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rescale_noisy_image(img):
    img = img + 0.5
    return img

def normalize_image_to_zero_one(img):
    img = img.clone()
    img -= img.min()
    img /= img.max()
    return img

def add_noise(noise,pic):
    noisy_pic = pic + noise
    return noisy_pic

def mse_to_psnr(mse):
    if isinstance(mse,float):
        psrn = 10 * np_log(1./mse)[0]/np_log(10)[0]
    else:
        psrn = 10 * np_log(1./mse)/np_log(10)
    return psrn

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

def write_pickle(data,fn,verbose=False):
    if verbose or settings.verbose >= 2: print("Writing pickle to [{}]".format(fn))
    with open(fn,'wb') as f:
        pickle.dump(data,f)
    if verbose: print("Save successful.")

def read_pickle(fn,verbose=False):
    if verbose or settings.verbose >= 2: print("Reading pickle file [{}]".format(fn))
    data = None
    with open(fn,'rb') as f:
        data = pickle.load(f)
    if verbose: print("Load successful.")
    return data

def get_model_epoch_info(cfg):
    if cfg.load:
        return 0,cfg.epoch_num+1
    else: return 0,0


def write_cfg(cfg,fpath):
    with io.open(fpath,'w',encoding='utf8') as f:
        yaml.dump(cfg,f,default_flow_style=False,allow_unicode=True)
        
def read_cfg(fpath):
    with open(fpath,'r') as f:
        cfg = yaml.load(f,Loader=yaml.Loader)
    return cfg

