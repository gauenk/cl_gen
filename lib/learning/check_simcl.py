
# python imports
import copy
from easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from apex import amp
from apex.parallel import DistributedDataParallel as apex_DDP
from torch.nn.parallel import DistributedDataParallel as th_DDP


# project imports
import settings
from datasets import load_dataset
# from .train import thtrain_simcl_cls as train_loop
from learning.train import thtrain_cls as train_loop
# from .test import thtest_simcl_cls as test_loop
from learning.test import thtest_cls as test_loop

# ------------------------------------------------------------------------
#  Testing Contrastive Learning => training a classifier using embeddings
# ------------------------------------------------------------------------


#
# -- load logit model --
#

class LogitCls(nn.Module):

    def __init__(self,cfg):
        super(LogitCls,self).__init__()
        # self.model = nn.Sequential(nn.Linear(3*32*32,cfg.dataset.n_classes))
        self.model = nn.Sequential(nn.Linear(cfg.enc_size,cfg.dataset.n_classes))

    def forward(self,x):
        return self.model(x)

def load_simcl_logit(cfg):
    logit = LogitCls(cfg)
    logit = logit.to(cfg.device)
    # if cfg.use_apex:
    #     logit = apex_DDP(logit)
    optimizer = optim.Adam(logit.parameters(),lr=3e-4)
    # rlrp = optim.lr_scheduler.ReduceLROnPlateau
    # scheduler = rlrp(optimizer,patience = 10,factor=1./np.sqrt(10))
    return logit,optimizer

def get_simcl_cfg(cfg):
    simcl_cfg = edict()

    simcl_cfg.device = cfg.device #'cuda:2'
    simcl_cfg.dataset = copy.deepcopy(cfg.dataset)
    simcl_cfg.proj_size = cfg.proj_size
    simcl_cfg.enc_size = cfg.enc_size

    # -- test for correctness -- 
    # simcl_cfg.dataset = edict()
    # simcl_cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    # simcl_cfg.dataset.n_classes = 10
    # simcl_cfg.dataset.name = "mnist"

    simcl_cfg.dataset.download = False
    simcl_cfg.num_workers = cfg.num_workers
    simcl_cfg.batch_size = 256
    simcl_cfg.rank = cfg.rank #2

    simcl_cfg.use_apex = False
    simcl_cfg.use_collate = False
    simcl_cfg.use_ddp = False
    simcl_cfg.world_size = 1

    simcl_cfg.global_step = 0
    simcl_cfg.log_interval = 20
    simcl_cfg.cls = simcl_cfg
    return simcl_cfg

def get_loader_features(loader,simcl):
    features = []
    targets = []
    for step, (imgs,target) in enumerate(loader):

        imgs = imgs.to(simcl.device)
        imgs = imgs.unsqueeze(1)

        with torch.no_grad():
            h,proj = simcl(imgs)
        h = torch.squeeze(h.cpu().detach().float())
        features.extend(h.numpy())
        targets.extend(target.numpy())
        if step % 25 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    features = np.array(features)
    targets = np.array(targets)
    print("Feature shape {}".format(features.shape))
    return features,targets

def get_features(simcl, loaders):
    X,y = edict(),edict()
    X.tr,y.tr = get_loader_features(loaders.tr,simcl)
    X.val,y.val = get_loader_features(loaders.val,simcl)
    X.te,y.te = get_loader_features(loaders.te,simcl)
    return X,y

def get_new_loaders(simcl, cls_loaders, batch_size):
    X,y = get_features(simcl, cls_loaders)
    loaders = edict()

    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X.tr),torch.from_numpy(y.tr)
    )
    train.data = torch.from_numpy(X.tr)
    loaders.tr = torch.utils.data.DataLoader(
        train, batch_size, shuffle=True, pin_memory=True
    )
        
    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X.val),torch.from_numpy(y.val)
    )
    val.data = torch.from_numpy(X.val)
    loaders.val = torch.utils.data.DataLoader(
        val, batch_size, shuffle=True, pin_memory=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X.te),torch.from_numpy(y.te)
    )
    test.data = torch.from_numpy(X.te)
    loaders.te = torch.utils.data.DataLoader(
        test, batch_size, shuffle=True, pin_memory=True
    )

    return loaders
    


#
# -- train/test a logit using contrastive learning repr --
#

def thtest_simcl(cfg, model, test_set):

    # remove ddp from model 
    if isinstance(model,th_DDP) or isinstance(model,apex_DDP):
        model = model.module
    model.eval()

    # get new config
    simcl_cfg = get_simcl_cfg(cfg)
    
    # get new data & loader
    data,cls_loaders = load_dataset(simcl_cfg,'cls')
    loaders = get_new_loaders(model, cls_loaders, simcl_cfg.batch_size)

    # load the logit
    logit,optimizer = load_simcl_logit(simcl_cfg)
    criterion = nn.CrossEntropyLoss()

    # apply apex
    if simcl_cfg.use_apex:
        logit, optimizer = amp.initialize(logit, optimizer, opt_level='O2')

    # train the logit
    print("Testing SimCl via Training",flush=True)
    n_epochs = 500
    for epoch in range(n_epochs):
        print(f"train_epoch: {epoch}",flush=True)
        train_loop(simcl_cfg, loaders.tr, logit, criterion,
                    optimizer, epoch, None)
        # train_loop(simcl_cfg, loaders.tr, logit, model, criterion,
        #            optimizer, epoch)
        if epoch % 25 == 0 and epoch > 0:
            loss = test_loop(simcl_cfg, logit, loaders.val)
            # loss = test_loop(simcl_cfg, model, logit, loaders.val)
            print(f"val loss @ [{epoch}]: {loss}")
            

    print("Testing linear classifier")
    # test logit
    if test_set == "val":
        loss = test_loop(simcl_cfg, logit, loaders.val)
        # loss = test_loop(simcl_cfg, model, logit, loaders.val)
    elif test_set == "test":
        loss = test_loop(simcl_cfg, logit, loaders.te)
        # loss = test_loop(simcl_cfg, model, logit, loaders.te)
    else:
        raise ValueError(f"Unknown data split [{test_set}]")
    return loss



