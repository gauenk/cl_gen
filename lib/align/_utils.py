
import torch
import numpy as np
import torch.nn.functional as F


def compute_epe(tensor_a,tensor_b):
    # nimages,npix,nframes-1,two
    tensor_a = tensor_a.cpu().type(torch.float)
    tensor_b = tensor_b.cpu().type(torch.float)
    epe = F.mse_loss(tensor_a,tensor_b,reduce='none')
    epe = torch.mean(epe,dims=(1,2,3))
    return epe

def torch_to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    else:
        return tensor

def construct_return_dict(fields,options):
    results = {}
    for field in fields:
        results[field] = options[field]
    return results


def check_all_str(py_list):
    return np.all([isinstance(e,str) for e in py_list])


def assert_cfg_fields(cfg):

    assert 'return_fields' in cfg, "Return fields required to modify return parameters"
    assert isinstance(cfg.return_fields,list), "Return fields must be a list"
    assert len(cfg.return_fields) > 0, "Return fields must be a non-empty"
    assert check_all_str(cfg.return_fields), "Return fields type must all be a str"



    return True


