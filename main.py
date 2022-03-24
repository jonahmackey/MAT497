import torch
import torch.nn as nn
from dataset.socal import SOCAL
from torch.utils.data import DataLoader
from net.aasp_model import AASP_Model
import json
import pandas as pd
import torch.optim as optim
from utils.lr_scheduler import WarmUpLR
from utils.meters import AverageMeter

if __name__ == "__main__":
    a = AverageMeter()
    
    a.add(0.8)
    
    print(a.val)
    print(a.sum)
    print(a.n)
    print(a.avg)