
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

from pyutils import images_to_psnrs

def compute_aligned_psnr(aligned_a,aligned_b,csize):
    nframes = aligned_a.shape[0]
    crop_a = tvF.center_crop(aligned_a,(csize.h,csize.w))
    crop_b = tvF.center_crop(aligned_b,(csize.h,csize.w))
    psnrs = []
    for t in range(nframes):
        batch_a = crop_a[t]
        batch_b = crop_b[t]
        psnr = images_to_psnrs(batch_a,batch_b)
        psnrs.append(psnr)
    psnrs = np.stack(psnrs,axis=0)
    return psnrs

def compute_epe(tensor_a,tensor_b):
    # nimages,npix,nframes-1,two
    tensor_a = tensor_a.cpu().type(torch.float)
    tensor_b = tensor_b.cpu().type(torch.float)
    epe = F.mse_loss(tensor_a,tensor_b,reduction='none')
    dims = torch.arange(tensor_a.ndim)
    epe = torch.mean(epe,dim=(1,3)).T
    return epe

def compute_pair_flow_acc(guess,gt):
    nimages,npix,nframes_m1,two = guess.shape
    guess = guess.cpu()
    gt = gt.cpu()
    flow_acc = torch.zeros(nframes_m1,nimages)
    for t in range(nframes_m1):
        guess_t = guess[:,:,t,:].type(torch.long)
        gt_t = gt[:,:,t,:].type(torch.long)
        both = torch.all(guess_t == gt_t,dim=-1)
        ncorrect = torch.sum(both,dim=1).type(torch.float)
        acc = 100 * ncorrect / npix
        flow_acc[t,:] = acc
    return flow_acc 

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


class BatchIter():
    def __init__(self,nsamples,batchsize):
        self.nsamples = nsamples
        self.batchsize = batchsize
        self.nbatches = nsamples // batchsize
        self.nbatches += (nsamples % batchsize) > 0
        self._iter_index = 0

    def __len__(self):
        return self.nbatches
    
    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= self.nbatches:
            raise StopIteration
        else:
            result = self[self._iter_index]
            self._iter_index += 1
            return result

    def __getitem__(self,index):
        start = index * self.batchsize
        end = start + self.batchsize
        pbatch = slice(start,end)
        return pbatch

