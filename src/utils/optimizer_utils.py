# Make scheduled optimizer
# https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/3
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


# Count parameters
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
