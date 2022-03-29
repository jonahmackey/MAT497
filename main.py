import torch
import torch.nn as nn
import torchvision
from dataset.socal import SOCAL
from torch.utils.data import DataLoader
from net.aasp_model import AASP_Model
import json
import pandas as pd
import torch.optim as optim
from utils.lr_scheduler import WarmUpLR
from utils.meters import AverageMeter
from collections import OrderedDict

if __name__ == "__main__":
    
    test_data = SOCAL(frame_res=64, train=False, dataset_path="../../socal")
    train_data = SOCAL(frame_res=64, train=True, dataset_path="../../socal")
    
    print(len(test_data))
    print(len(train_data))
    
    
    