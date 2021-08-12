import os
import torch

def resume_training(argdict, model, optimizer):
    """ 
    Resumes previous training or starts anew

    """
    if argdict['resume_training']:
        resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = argdict['epochs']
            new_milestone = argdict['milestone']
            current_lr = argdict['lr']
            argdict = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            argdict['epochs'] = new_epoch
            argdict['milestone'] = new_milestone
            argdict['lr'] = current_lr
            print("=> loaded checkpoint '{}' (epoch {})"\
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = checkpoint['args']
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))
    
            argdict['resume_training'] = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))
    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = argdict['no_orthog']

    return start_epoch, training_params

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
    """Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
    Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
    """
    torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
    save_dict = { \
        'state_dict': model.state_dict(), \
        'optimizer' : optimizer.state_dict(), \
        'training_params': train_pars, \
        'args': argdict\
        }
    torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

    if epoch % argdict['save_every_epochs'] == 0:
        torch.save(save_dict, os.path.join(argdict['log_dir'],
                                           'ckpt_e{}.pth'.format(epoch+1)))
    del save_dict

