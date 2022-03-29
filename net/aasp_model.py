import torch
import torch.nn as nn
import torchvision
from .resnet import resnet18, resnet34, resnet50
from .transformer import TransformerEncoderLayer, TransformerEncoder, PositionalEncoding

from collections import OrderedDict


class AASP_Model(nn.Module):
    def __init__(self, enc_model: str, enc_norm: str, pretrained: bool, freeze_base: bool, num_layers: int, num_heads: int, 
                 embed_dim: int, norm_first: bool, pe: bool, dropout: float):
        """AASP (Automatic Assessment of Neurosurgical Performance) Model.

        Args:
            enc_model (str): image encoder model architecture (resnet18, resnet34, resnet50)
            enc_norm (str): image encoder norm layer (BN, LN, IN)
            num_layers (int): number of transformer encoder layers
            num_heads (int): number of attention heads
            embed_dim (int): embedding dimension of transformer
            norm_first (bool): Pre-LN or Post-LN
            pe (bool): apply positional encoding
            dropout (float): dropout probability in transformer
        """
        super(AASP_Model, self).__init__()
        
        self.enc_model = enc_model
        self.enc_norm = enc_norm
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_first = norm_first
        self.pe = pe
        self.dropout = dropout
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        
        if enc_model == 'resnet18':
            if pretrained:
                weights = torch.load("../../../model_weights/resnet18-f37072fd.pth")
                resnet = torchvision.models.resnet18(pretrained=False) 
                resnet.load_state_dict(weights)
                
                self.features = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2])) 
                
                if freeze_base:
                    for layer in self.features.parameters():
                        layer.requires_grad = False
            else:
                self.features = resnet18(norm=enc_norm)
                
        elif enc_model == 'resnet34':
            if pretrained:
                weights = torch.load("../../../model_weights/resnet34-b627a593.pth")
                resnet = torchvision.models.resnet34(pretrained=False) 
                resnet.load_state_dict(weights)
                
                self.features = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2])) 
                
                if freeze_base:
                    for layer in self.features.parameters():
                        layer.requires_grad = False
            else:
                self.features = resnet34(norm=enc_norm)
                
        elif enc_model == 'resnet50':
            if pretrained:
                weights = torch.load("../../../model_weights/resnet50-0676ba61.pth")
                resnet = torchvision.models.resnet50(pretrained=False) 
                resnet.load_state_dict(weights)
            
                self.features = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2]))
                
                if freeze_base:
                    for layer in self.features.parameters():
                        layer.requires_grad = False
            else:
                self.features = resnet50(norm=enc_norm)
        else:
            raise Exception('enc_model must be resnet18, resnet34, or resnet50!')  
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.projection = None
        if enc_model == 'resnet50':
            self.projection = nn.Linear(in_features=2048, out_features=embed_dim) 
        
        if pe:
            self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout, max_len=61)

        self.cls_token = nn.Parameter(data=torch.rand((1, 1, embed_dim), dtype=torch.float32), requires_grad=True)

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, 
                                                   batch_first=True, norm_first=norm_first)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, 
                                                 norm=nn.LayerNorm(normalized_shape=embed_dim))

        self.classifier = nn.Linear(in_features=embed_dim, out_features=1)

    def forward(self, x):
        """
        Input shape = (B, S, C, H, W), where
        - B = batch size = 1
        - S = sequence length = 60
        - C = input channels = 3
        - H = height = 224
        - W = width = 224
        """
        
        # get features
        x = self.features(x.squeeze()) # shape = (S, C_out, 7, 7)
        x = self.global_avg_pool(x).squeeze() # shape = (S, C_out)
        
        # apply projection
        if self.projection is not None:
            x = self.projection(x) # shape = (S, E)
        x = x.unsqueeze(dim=0) # shape = (1, S, E)

        # add CLS token
        x = torch.cat((self.cls_token, x), dim=1) # shape = (1, S', E), S' = S + 1

        # apply positional encoding
        if self.pe:
            x = self.positional_encoding(x) # shape = (1, S', E)
            
        # transformer
        x = self.transformer(x).squeeze() # shape = (S', E) 
        
        # make prediction using cls token
        out = self.classifier(x[0])

        return out
