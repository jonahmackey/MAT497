import torch
import torch.nn as nn
import math
from torchvision.models import resnet50, vgg19
from collections import OrderedDict


class AASP_Model(nn.Module):
  def __init__(self, pretrained_resnet=True, embed_dim=512, nhead=8, num_layers=6, dropout=0.1):
    """
    embed_dim = feature dimension
    nhead = number of heads in multihead attention
    num_layers = number of stacked transformer encoder layers
    """
    super(AASP_Model, self).__init__()

    resnet = resnet50(pretrained=pretrained_resnet) # get resnet50
    
    self.resnet_base = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2])) # remove head

    self.global_avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=7)) # global average pool

    self.projection = nn.Linear(in_features=2048, out_features=embed_dim) # add relu before/after to make nonlinear projection?

    self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout, max_len=61)

    self.cls_token = nn.Parameter(data=torch.rand((1, 1, embed_dim), dtype=torch.float32)) # make CLS token, shape = 1x1xE *** (req grad=True??)

    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
    self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                             num_layers=num_layers, 
                                             norm=nn.LayerNorm(normalized_shape=embed_dim)) # transformer encoder

    self.classifier = nn.Linear(in_features=embed_dim, out_features=1)

  def forward(self, X):
    """
    Parameters:
      x =  Tensor with shape (B, S, C, H, W)

    B = batch size = 1
    S = sequence length <= 60
    C = in channels = 3
    H = height = 224
    W = width = 224 
    """
    # input shape = (1, S, 3, 224, 224)

    # print X.mean and X.std across dims (0, 1, 3, 4), we want mean = 0, std = 1
    # print X.shape
    
    # get features
    X = self.resnet_base(X.squeeze()) # shape = (S, 2048, 7, 7)
    X = self.global_avg_pool(X).squeeze() # shape = (S, 2048)
    X = self.projection(X).unsqueeze(dim=0) # shape = (1, S, E)

    # add CLS
    X = torch.cat((self.cls_token, X), dim=1) # shape = (1, S1, E), S1 = S + 1

    # transformer
    X = self.positional_encoding(X) # shape = (1, S1, E)
    X = self.transformer(X).squeeze() # shape = (S1, E) ***
    
    # ebl prediction
    ebl_prediction = self.classifier(X[0])

    return ebl_prediction


class AASP_Model_S(nn.Module):
  def __init__(self, embed_dim=512, nhead=8, num_layers=6, dropout=0.1):
    """
    embed_dim = feature dimension
    nhead = number of heads in multihead attention
    num_layers = number of stacked transformer encoder layers
    """
    super(AASP_Model_S, self).__init__()

    vgg = vgg19(pretrained=False)
    self.resnet_base = nn.Sequential(OrderedDict(list(vgg.named_children())[:-2]))

    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    # self.projection = nn.Linear(in_features=512, out_features=embed_dim) # add relu before/after to make nonlinear projection?

    # self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout, max_len=61)

    self.cls_token = nn.Parameter(data=torch.rand((1, 1, embed_dim), dtype=torch.float32), requires_grad=True)

    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
    self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                             num_layers=num_layers, 
                                             norm=nn.LayerNorm(normalized_shape=embed_dim)) # transformer encoder

    self.classifier = nn.Linear(in_features=embed_dim, out_features=1)

  def forward(self, X):
    # input shape = (1, S, 3, H, W)

    # get features
    X = self.resnet_base(X.squeeze()) # shape = (S, 512, 2, 2)
    X = self.global_avg_pool(X).squeeze() # shape = (S, 512)
    # X = self.projection(X).unsqueeze(dim=0) # shape = (1, S, 512)
    X = X.unsqueeze(dim=0) # (1, S, 512)

    # add CLS
    X = torch.cat((self.cls_token, X), dim=1) # shape = (1, S1, 512), S1 = S + 1

    # transformer
    # X = self.positional_encoding(X) # shape = (1, S1, 512)
    X = self.transformer(X).squeeze() # shape = (S1, 512) ***
    
    # out prediction
    X = self.classifier(X[0])

    return X
  
  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (61, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (512)
        pe = torch.zeros(1, max_len, d_model) # (1, 61, 512)
        pe[0, :, 0::2] = torch.sin(position * div_term) # (1, 61, 512)
        pe[0, :, 1::2] = torch.cos(position * div_term) # (1, 61, 512)
        self.register_buffer('pe', pe)

    def forward(self, X):
        """
        Parameters:
            X = Tensor with shape [batch_size, seq_len, embedding_dim]
        """
        X = X + self.pe[0, :X.size(1)]
        return self.dropout(X)