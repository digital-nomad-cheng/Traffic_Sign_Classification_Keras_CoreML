def create_lr_schedule(epochs, lr_base, lr_power=0.99, mode='power_decay'):
    return lambda epoch: _lr_schedule(epoch, epochs, lr_base, lr_power, mode)


def _lr_schedule(epoch, epochs, lr_base, lr_power, mode):
    if mode is 'power_decay':
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    if mode is 'adam':
        lr = 0.001
    if mode is 'progressive_drops':
        """
        if epoch > 0.9 * epochs:
            lr = 0.00001
        elif epoch > 0.75 * epochs:
            lr = 0.00005
        elif epoch > 0.5 * epochs:
            lr = 0.0001
        elif epoch > 0.2 *epochs:
            lr = 0.0005
        elif epoch > 25:
            lr = 0.001
        else:
            lr = 0.05
        """
        if epoch > 75:
            lr = 0.3
        elif epoch > 50:
            lr = 0.5
        elif epoch > 25:
            lr = 0.7
        else:
            lr = 1.0
    print('lr: %f' % lr)

    return lr

