"""
Test the model
"""

# python imports
from tqdm import tqdm

# torch imports
import torch as th
import torch.nn.functional as F

# project imports
from pyutils.misc import np_log,rescale_noisy_image,mse_to_psnr
from layers.denoising import reconstruct_set

def thtest_cls(args, model, test_loader):
    # test a classifier
    device = args.device
    model.eval()
    test_loss = 0
    correct = 0
    with th.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.long()#.squeeze_()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,correct,len(test_loader.dataset)


def thtest_static(args, enc, dec, test_loader, use_psnr=False):
    # test a denoising task
    device = args.device
    enc.eval()
    dec.eval()
    test_loss = 0
    correct = 0

    idx = 0
    with th.no_grad():
        for pic_set, th_img in tqdm(test_loader):
            set_loss = 0
            th_img = th_img.to(device)
            pic_set = pic_set.to(device)

            N = len(pic_set)
            BS = len(pic_set[0])
            pshape = pic_set[0][0].shape
            shape = (N,BS,) + pshape

            rec_set = reconstruct_set(pic_set,enc,dec,args.share_enc)
            rec_set = rescale_noisy_image(rec_set)
            cmp_img = th_img.expand(shape)
            set_loss = F.mse_loss(cmp_img,rec_set).item()
            if use_psnr: set_loss = mse_to_psnr(set_loss)
            test_loss += set_loss
            idx += 1
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:2.3e}\n'.format(test_loss))
    return test_loss



def thtest_denoising(cfg, model, test_loader):
    model.eval()
    test_loss = 0

    idx = 0
    with th.no_grad():
        for noisy_imgs, raw_img in tqdm(test_loader):
            set_loss = 0
            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            raw_img = raw_img.cuda(non_blocking=True)

            noisy_imgs = noisy_imgs.cuda(non_blocking=True)
            dec_imgs,proj = model(noisy_imgs)
            dec_imgs = rescale_noisy_image(dec_imgs)

            N = len(dec_imgs)
            BS = len(dec_imgs[0])
            dshape = (N,BS,) + dec_imgs.shape[2:]

            dec_imgs = dec_imgs.reshape(dshape)
            raw_img = raw_img.expand(dshape)
            loss = F.mse_loss(raw_img,dec_imgs).item()
            if cfg.test_with_psnr: loss = mse_to_psnr(loss)
            test_loss += loss
            idx += 1
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:2.3e}\n'.format(test_loss))
    return test_loss



