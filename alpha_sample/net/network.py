import torch.nn as nn
import torch.nn.functional as F

class Network:
    def construct(self, net, obj):
        targetClass = getattr(self, net)
        instance = targetClass(obj)
        return instance
       
    class MLP(nn.Module):
        def __init__(self,obj):
            super(Network.MLP, self).__init__()
            
            num_layers = 3
            
            self.fc_list   = nn.ModuleList()
            self.relu_list = nn.ModuleList()
            
            if hasattr(obj, 'net_width'):
                net_width = obj.net_width
            else:
                net_width = 1024
            
            for i in range(num_layers-1):
                if i == 0:
                    self.fc_list.append(nn.Linear(obj.padded_im_size**2 * obj.input_ch,
                                                  net_width))
                else:
                    self.fc_list.append(nn.Linear(net_width, net_width))

                self.relu_list.append(nn.ReLU(inplace=True))
            
            if num_layers == 1:
                self.classifier = nn.Linear(obj.padded_im_size**2 * obj.input_ch,
                                            obj.num_classes)
            else:
                self.classifier = nn.Linear(net_width, obj.num_classes)
            
            
        def forward(self, x):
            x = x.view(x.shape[0], -1)

            out = x
            
            for i in range(len(self.relu_list)):
                out = self.relu_list[i](self.fc_list[i](out))

            out = self.classifier(out)
            
            return out
        
        
    class LeNet(nn.Module):
        def __init__(self, obj):
            super(Network.LeNet, self).__init__()
            sz = int(((obj.padded_im_size - 2*2)/2 - 2*2) / 2)

            self.conv1 = nn.Conv2d(obj.input_ch, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            
            self.fc1   = nn.Linear(16*sz**2, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, obj.num_classes)
            
            self.relu1 = nn.ReLU(inplace=False)
            self.relu2 = nn.ReLU(inplace=False)
            self.relu3 = nn.ReLU(inplace=False)
            self.relu4 = nn.ReLU(inplace=False)
            
        def forward(self, x):
            out = self.relu1(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = self.relu2(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.relu3(self.fc1(out))
            out = self.relu4(self.fc2(out))
            out = self.fc3(out)
            return out
        
        