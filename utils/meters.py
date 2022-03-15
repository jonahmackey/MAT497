import torch
import math
import numpy as np


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.

    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self):
        '''Log a new value to the meter.'''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass
    
    
class MSEMeter(Meter):
    def __init__(self, root=False):
        super(MSEMeter, self).__init__()
        self.reset()
        self.root = root

    def reset(self):
        self.n = 0
        self.sesum = 0

    def add(self, output, target, n=1):
        self.n += n
        self.sesum += ((output - target) ** 2).sum().item()

    def value(self):
        mse = self.sesum / max(1, self.n)
        return math.sqrt(mse) if self.root else mse
      
    
class AccuracyMeter(Meter):
    def __init__(self):
        super(AccuracyMeter, self).__init__()
        self.reset()
    
    def add(self, output, target, n=1):
        pred = (output >= 0.5).type(torch.int)
        self.correct += (pred == target).type(torch.int).sum().item()
        self.n += n
      
    def value(self):
      if self.n == 0:
        return np.nan
      
      else:
        return 100 * (self.correct / self.n)
      
    def reset(self):
      self.correct = 0
      self.n = 0
      
      
class AverageMeter(Meter):
    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.n += n
        self.avg += self.sum / self.count
        
    def value(self):
        return self.avg

    def reset(self):
        self.n = 0
        self.sum = 0
        self.val = 0
        self.avg = 0
      