from bisect import bisect_right
from torch.optim.optimizer import Optimizer


class WarmUpLR(object):
    """ Set initial learning rate to zero and increase linearly over t_warmup 
    steps to lr_max, then decrease the learning rate proportionally to the
    inverse square root of step number. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_max (float): Max learning rate acheived after warmup stage.
        t_warmup (int): Number of warm up steps.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_max, t_warmup):
        super(WarmUpLR, self).__init__()
        
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        
        optimizer.param_groups[0].setdefault('initial_lr', optimizer.param_groups[0]['lr'])
    
        self.step(0)
        self.last_iter = 0
        
        self.lr_max = lr_max
        self.t_warmup = t_warmup

    def get_lr(self):
        if self.last_iter <= self.t_warmup:    
            return self.lr_max * (self.last_iter * (self.t_warmup ** (-1.5)))
        else: 
            return self.lr_max * (self.last_iter ** (-0.5)) 
    
    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        
        self.optimizer.param_groups[0]['lr'] = self.get_lr()
        
