"""
Test the model
"""

# python imports
from tqdm import tqdm


# torch imports
import torch as th
import torch.nn.functional as F

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


def thtest_static(args, enc, dec, test_loader):
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
            for x in pic_set:
                x = x.to(device)
                h,aux = enc(x)
                r = dec([h,aux])
                set_loss_i = F.mse_loss(th_img,r).item()
                set_loss += set_loss_i
            set_loss /= len(pic_set)
            test_loss += set_loss
            idx += 1
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


