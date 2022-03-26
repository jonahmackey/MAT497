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
from torchvision.models.video import r2plus1d_18

if __name__ == "__main__":
    a = AverageMeter()
    
    a = torch.rand(1, 3, 60, 112, 112)
    
    model = r2plus1d_18(pretrained=True)
    print(model.state_dict)
    model.eval()
    
    out = model(a)
    print(out.shape)