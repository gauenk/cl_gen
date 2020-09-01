"""
Training for Pytorch

"""

import torch
import torch.nn.functional as F


def thtrain_cls(cfg, train_loader, model, criterion, optimizer, epoch, writer):
    # train a classifier
    model.train()
    idx = 0
    loss_epoch = 0
    print("N samples:", len(train_loader.dataset.data))
    for batch_idx, (data, target) in enumerate(train_loader):
        idx += cfg.batch_size

        optimizer.zero_grad()
        data, target = data.to(cfg.device), target.to(cfg.device)
        target = target.long()

        output = model(data)
        loss = F.nll_loss(output, target)
        #loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train_epoch", loss.item(), cfg.global_step)
        cfg.global_step += 1
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss_epoch

def thtrain_cl(cfg, train_loader, model, criterion, optimizer, epoch, writer):
    # train contrastive learning
    model.train()
    idx = 0
    loss_epoch = 0
    data = train_loader.dataset.data
    print("N samples:", len(data))
    for batch_idx, ((x_i,x_j), _) in enumerate(train_loader):
        idx += cfg.batch_size

        optimizer.zero_grad()
        x_i,x_j = x_i.to(cfg.device),x_j.to(cfg.device)

        # forward step & loss
        h_i,h_j,z_i,z_j = model(x_i,x_j)
        loss = criterion(z_i,z_j)
        loss.backward()
        optimizer.step()

        # print updates
        writer.add_scalar("Loss/train_epoch", loss.item(), cfg.global_step)
        cfg.global_step += 1
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * cfg.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        loss_epoch += loss.item()
    return loss_epoch


def thtrain_imgrec(cfg, train_loader, transforms, model, criterion, optimizer, epoch, writer):
    # todo: dataset for (xtrans, transforms, target)
    # todo: model for (x_trans, trasnforms)

    # train contrastive learning
    model.train()
    idx = 0
    loss_epoch = 0
    data = train_loader.dataset.data
    print("N samples:", len(data))
    transforms = transforms.to(cfg.device)
    for batch_idx, (transformed_images, data_idx) in enumerate(train_loader):

        # setup the forward pass
        idx += cfg.batch_size
        optimizer.zero_grad()
        
        # preprocess the data; on-the-fly right now. can change to pre-process later
        transformed_images = [x.to(cfg.device) for x in transformed_images]
        encoded_images = [criterion.encoder(x) for x in transformed_images]
        projected_images = [criterion.projector(x) for x in encoded_images]
        batch_transforms = get_batch_transforms(transforms,data_idx)


        # -- forward pass --
        # [encoded_images] NumTransforms x BatchSize x 2048 (EncSize)
        # [transformed_images] NumTransforms x BatchSize x ColorChan x ImgH x ImgW
        # [recon_images] NumTransforms x BatchSize x ColorChan x ImgH x ImgW
        reconstructed_imgs = model(encoded_images)
        
        # loss function
        loss = criterion(reconstructed_imgs, projected_images, batch_transforms)
        loss.backward()
        optimizer.step()

        # print updates
        writer.add_scalar("Loss/train_epoch", loss.item(), cfg.global_step)
        cfg.global_step += 1
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * cfg.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        loss_epoch += loss.item()
    return loss_epoch

def get_batch_transforms(transforms,data_idx):
    trans = []
    for idx in data_idx:
        trans.append(transforms[idx])
    return trans


def thtrain_disent(cfg, train_loader, models, criterion, optimizer, epoch, writer):

    for name,model in models.items(): model.train()
    idx = 0
    loss_epoch = 0
    data = train_loader.dataset.data
    print("N samples:", len(data))
    for batch_idx, (transformed_imgs, th_img) in enumerate(train_loader):

        # setup the forward pass
        idx += cfg.batch_size
        optimizer.zero_grad()
        
        # preprocess the data; on-the-fly right now. can change to pre-process later
        t_imgs = [x.to(cfg.device) for x in transformed_imgs]
        loss = criterion(t_imgs)
        loss.backward()
        optimizer.step()

        # print updates
        writer.add_scalar("Loss/train_epoch", loss.item(), cfg.global_step)
        cfg.global_step += 1
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * cfg.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        loss_epoch += loss.item()
    return loss_epoch


