import os
import torch


def save_checkpoint(model, bast_acc, val_acc, epoch, path='./checkpoint/ckpt_'):
    # Save checkpoint.
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, path + str(0) + '.pth')

    if bast_acc == None:
        return val_acc
    elif bast_acc > val_acc:
        print('Saving..')
        torch.save(state, path + str(round(float(val_acc),6)) + '.pth')
        return val_acc
    return bast_acc

def load_checkpoint(model:torch.nn.Module,path='checkpoint/ckpt_0.pth'):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'],strict=False)
    start_epoch = checkpoint['epoch']
    return model