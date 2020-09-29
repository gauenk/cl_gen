
"""
A simple example to train simCLR on dataset A
"""

# python imports
import sys,os
sys.path.append("./lib/")
import numpy as np
from easydict import EasyDict as edict
import pathlib
from pathlib import Path

# torch imports
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


# project imports
from pyutils.cfg import get_cfg
from layers import NT_Xent,SimCLR,get_resnet,LogisticRegression,load_simclr
from learning.train import thtrain_cl as train_cl
from learning.train import thtrain_cls as train_cls
from learning.test import thtest_cls as test_cls
from learning.utils import save_model
from torchvision.datasets import CIFAR10
from datasets import ClCIFAR10,TransformsSimCLR,ImgRecCIFAR10,get_cifar10_dataset


def load_cls_for_simclr(cfg):
    simCLR,_,_ = load_simclr(cfg)
    logit = LogisticRegression(simCLR,cfg.cls.dataset.n_classes)
    if cfg.cls.load:
        model_fp = os.path.join(
            cfg.cls.model_path, "checkpoint_{}.tar".format(cfg.cls.epoch_num)
        )
        logit.load_state_dict(torch.load(model_fp, map_location=cfg.cls.device.type))
    cfg_adam = cfg.cls.optim.adam
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_adam.lr)  # TODO: LARS
    scheduler = None
    return logit,optimizer,scheduler

def train_model_cl(cfg):

    gpu = cfg.gpuid
    torch.cuda.set_device(gpu)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)    

    data,loader = get_dataset(cfg,"cl")
    # data,loader = get_cifar10_dataset(cfg,'cl')
    model,optimizer,scheduler = load_simclr(cfg)
    criterion = NT_Xent(cfg.cl.batch_size, cfg.cl.temperature, cfg.cl.device, cfg.world_size)

    writer = SummaryWriter()
    global_step,current_epoch = get_model_epoch_info(cfg.cl)
    cfg.cl.global_step = global_step
    cfg.cl.current_epoch = current_epoch
    for epoch in range(cfg.cl.current_epoch, cfg.cl.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_cl(cfg.cl, loader.tr, model, criterion, optimizer, epoch, writer)

        if scheduler:
            scheduler.step()

        if epoch % cfg.cl.checkpoint_interval == 0:
            save_model(cfg.cl, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.cl.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t lr: {round(lr, 5)}"
        )
        cfg.cl.current_epoch += 1

    save_model(cfg.cl, model, optimizer)

def get_model_epoch_info(cfg):
    if cfg.load:
        return 0,cfg.epoch_num+1
    else: return 0,0
    
def train_model_cls(cfg):

    # init some random seeds, misc
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)    

    # get latest model version
    # epoch_num = get_latest_model_version(cfg)

    # load model from saved point with SimCLR base encoding
    data,loader = get_cifar10_dataset(cfg,'cls')
    model,optimizer,scheduler = load_cls_for_simclr(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()

    # train logit model for cls
    global_step,current_epoch = get_model_epoch_info(cfg.cls)
    cfg.cls.global_step = global_step
    cfg.cls.current_epoch = current_epoch
    for epoch in range(cfg.cls.current_epoch,cfg.cls.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_cls(cfg.cls, loader.tr, model, criterion, optimizer, epoch, writer)

        if scheduler:
            scheduler.step()

        if epoch % cfg.cls.checkpoint_interval == 0:
            save_model(cfg.cls, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(loader.tr), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{cfg.cls.epochs}]\t Loss: {loss_epoch / len(loader.tr)}\t lr: {round(lr, 5)}"
        )
        cfg.cls.current_epoch += 1

    save_model(cfg.cls, model, optimizer)


def test_model_cls(cfg):

    print("Test Model on Classification")
    # init some random seeds, misc
    cfg.logit_reload = True
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)    

    # get latest model version
    # epoch_num = get_latest_model_version(cfg)

    # load model from saved point with SimCLR base encoding
    data,loader = get_cifar10_dataset(cfg,'cls')
    model,optimizer,scheduler = load_cls_for_simclr(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    # test on train data
    print("Testing on Training Data")
    testTr = edict()
    testTr.loss,testTr.correct,testTr.N = test_cls(cfg.cls, model, loader.tr)

    # test on test data
    print("Testing on Testing Data")
    testTe = edict()
    testTe.loss,testTe.correct,testTe.N = test_cls(cfg.cls, model, loader.te)

    print("Accuracy on Training Data:")
    print(testTr)

    print("Accuracy on Testing Data:")
    print(testTe)

    
def get_denoising_criterion():
    return torch.nn.L1Loss()
    # return torch.nn.MSELoss()
    # return torch.nn.CosineSimilarity(dim=1,eps=1e-6)

def test_grad_image(cfg):
    """
    Testing code to use the gradient to update the input image

    1. get a raw image embedding; f(x_raw)
    2. get a modified image embedding; f(x_modified), x_modified = transform(x_raw)
    3. compute loss between embeddings to update the modified image; L(f_raw,f_mod)
    4. check if new image matches the old one; plot me and compute l2-norm
    """
    print("Starting Image Gradient Test")
    cfg.cl.dataset.transforms.low_light = True
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)    

    data,loader = get_cifar10_dataset(cfg,'cl')
    model,optimizer,scheduler = load_simclr(cfg)
    criterion = get_denoising_criterion()

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False    
    
    # run test
    niters = 100
    ilt = iter(loader.tr)
    for i in range(10):
        next(ilt)
    (img_mod,_),img_raw = next(ilt)
    print(img_mod.shape)
    print(img_raw.shape)    
    print(torch.norm(img_raw),torch.norm(img_mod))
    print(type(img_mod),type(img_raw))
    img_raw,img_mod = img_raw.to(cfg.cl.device),img_mod.to(cfg.cl.device)
    img_mod.requires_grad_(True)
    params = [img_mod]
    optimizer = torch.optim.LBFGS(params)
    # optimizer = torch.optim.Adam(params, lr=cfg.cl.optim.adam.lr)

    tol = 10e-8
    print_interval = 500
    save_interval = 1000

    loss = 100
    iters = 0
    # this optimization is VERY slow.
    # maybe our contribution can include a faster convergence scheme?
    save_image(img_raw,"img_raw.png")
    save_image(img_mod,"img_mod_0.png")
    run = [0]
    while(loss > tol):
        def closure():
            # correct the values of updated input image
            img_mod.data.clamp_(0, 1)
            optimizer.zero_grad()
            enc_raw,enc_mod,_,_ = model(img_raw,img_mod)
            _loss = criterion(enc_raw,enc_mod)
            _loss.backward()
            loss = _loss.item()

            run[0] += 1
            # if run[0] % 50 == 0:
            #     print("run {}:".format(run))
            #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            #         style_score.item(), content_score.item()))
            #     print()
            iters = run[0]

            if iters % print_interval == 0:
                print(f"Iteration: {iters}")
                print("Loss: {:2.3e}".format(loss))
                print("|| Img_mod || = {:.3f}".format(torch.norm(img_mod).item()))
                print("|| Enc(Img_mod) || = {:.3f}".format(torch.norm(enc_mod).item()))
                print("|| Img_raw || = {:.3f}".format(torch.norm(img_raw).item()))
                print("|| Enc(Img_raw) || = {:.3f}".format(torch.norm(enc_raw).item()))
    
            if iters % save_interval == 0:
                save_image(img_mod,f"img_mod_{iters}.png")
    
            return loss
        optimizer.step(closure)
        iters = run[0]

        # iters += 1
        # optimizer.zero_grad()
        # enc_raw,enc_mod,_,_ = model(img_raw,img_mod)
        # _loss = criterion(enc_raw,enc_mod)
        # _loss.backward()
        # loss = _loss.item()
        # optimizer.step()

        # if iters % print_interval == 0:
        #     print("Loss: {:2.3e}".format(loss))
        #     print("|| Img_mod || = {:.3f}".format(torch.norm(img_mod).item()))
        #     print("|| Enc(Img_mod) || = {:.3f}".format(torch.norm(enc_mod).item()))
        #     print("|| Img_raw || = {:.3f}".format(torch.norm(img_raw).item()))
        #     print("|| Enc(Img_raw) || = {:.3f}".format(torch.norm(enc_raw).item()))

        # if iters % save_interval == 0:
        #     save_image(img_mod,f"img_mod_{iters}.png")
            
    save_image(img_mod,f"img_mod_{iters}.png")
    print("DONE!")    
    
        

if __name__ == "__main__":


    cfg = get_cfg()


    # train simCLR
    cfg.gpuid = 0
    cfg.cl.load = False
    cfg.cl.epoch_num = 0
    cfg.cl.epochs = 1000
    cfg.cls.load = False
    train_model_cl(cfg)
    print("Contrastive Learning Training Complete")
    exit()

    # train logit using simCLR
    cfg.cl.load = True
    cfg.cl.epoch_num = 300
    cfg.cls.load = True
    cfg.cls.epoch_num = 20
    cfg.cls.epochs = 40
    cfg.cls.model_path = Path(f"./output/cls/{cfg.exp_name}/{cfg.cl.epoch_num}")
    if not cfg.cls.model_path.exists(): cfg.cls.model_path.mkdir()
    # train_model_cls(cfg)
    print("Classifier Training Complete")

    # test logit
    cfg.cl.load = True
    cfg.cl.epoch_num = 300
    cfg.cls.load = True
    cfg.cls.epoch_num = 40
    # test_model_cls(cfg)
    print("Classifier Testing Complete")

    # run input image grad test
    # exit()
    cfg.cl.batch_size = 1
    cfg.cl.load = True
    cfg.cl.epoch_num = 300
    cfg.cls.load = True
    cfg.cls.epoch_num = 10
    test_grad_image(cfg)
    print("Grad Image Test Complete")
    
"""
pair -> set

64 images of background (static) + car (moving)
difference in car (since moving)

can we decouple the background and the car?
can we reconstruct a single frame well?
can we capture how the car moves in the embeddings?
how to actually do denoising from Neurips?
first part, train a bunch of individual encoders...
design encoders in such a way such taht the first couple elements are common
fix the embeddings for some number; 
"information disentaglement"

0. read the Neurips paper more carfully to do the denoising
   - re-implement results oyo
1. (setup) toy problem with no motion and just noise;
   - train contrastive learning network for a set of images (2 -> N)
   - if N = 10, how to define the loss function
   (exp) demonstrate as N grows, what is the benefit of the model?
   - plot for # of epochs v.s. accuracy of logit@40
   - if iid Gaussian noisy images, then taking N = 10 should be better than N = 2
   - plot # of input images v.s. reconstruction error
3. incorporate motion and use the "fixed encoder" scheme
   - e.g. frame 2: top 200 of 256 are fixed


we have a unsupervised learning algorithm for bursty image recovery
don't have to tie it too closly to QIS; just for iid Gaussian noise


"""
