import torch
import torch.nn as nn
from dataset.socal import SOCAL
from torch.utils.data import DataLoader
from net.aasp_model import AASP_Model
import json
import pandas as pd
import torch.optim as optim
from utils.lr_scheduler import WarmUpLR

if __name__ == "__main__":
    output =    '{}\tLoss: {} ({})\tEpoch: [{}/{}][{}/{}]'.format('Train'.capitalize(), 29, 0.6, 0.7, 129, 0.5, 200)
                    
    print(output)