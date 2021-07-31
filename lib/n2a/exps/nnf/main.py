
"""
Compare ability to recover nnf + of for each method.


"""

# -- python imports --
import numpy as np

# -- pytorch imports --
import torch

# -- project imports --
from n2a.align import compute_alignment
from n2a.exps._image import load_image_dataset

EXP_NAME = "nnf"

def execute_experiment(cfg):

    # -- init results --
    nnf_gt,nnf_acc,of_epe = [],[],[]

    # -- set random seed --
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # -- load data --
    image_data,batch_loaders = load_image_dataset(cfg,EXP_NAME)
    batch_iter = iter(batch_loaders.tr)

    # -- compute for several images --
    NUM_BATCHES = 100
    for batch_idx in range(NUM_BATCHES):

        # -- get batch --
        burst, gt_of, gt_nnf = next(batch_iter)
        
        # -- compute alignment --
        aligned,flow = compute_alignment(cfg,burst)

        # -- measure to groundtruth --
        nn_acc_i = compute_nnf_accuracy(gt_nnf,flow)
        of_epe_i = compute_of_epe(gt_of,flow)

        # -- save results --
        nnf_acc.append(nnf_acc_i)
        of_epe.append(of_epe_i)
        nnf_gt.append(gt_nnf)

    # -- format results --
    nnf_acc_mean = np.mean(nnf_acc)
    nnf_acc_stderr = np.std(nnf_acc)/len(nnf_acc)
    of_epe_mean = np.mean(of_epe)
    of_epe_stderr = np.std(of_epe)/len(of_epe)
    results = pd.DataFrame({'nnf-acc':nnf_acc_mean,'nnf-acc-stderr':nnf_acc_stderr,
                            'of-epe':of_epe_mean,'of-epe-std':of_epe_stderr,'nnf-gt':nnf_gt})
    return results
