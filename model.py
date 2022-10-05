import timm
from pprint import pprint
import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class DetHead(nn.Module):

    def __init__(self,layers_sizes,input_channel,kernel_size = (3,3)):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_channels = input_channel
        self.layer_size = layers_sizes
        for i in range(len(layers_sizes)):
            self.layers.append(nn.Conv2d(in_channels=curr_channels,out_channels=layers_sizes[i],kernel_size=kernel_size,stride=(1,1),padding=(1,1)))
            curr_channels=layers_sizes[i]

    def forward(self,features):
        inputs = features
        for i in range(len(self.layers)):
            inputs = nn.BatchNorm2d(self.layer_size[i],device='cuda')(F.relu(self.layers[i](inputs)))
        return inputs

class BackBone(nn.Module):

    def __init__(self,input_channel=64):
        super().__init__()

        self.dla_model = timm.create_model('dla34',features_only=True,pretrained=True)
        hm_list = [60,20,3]
        self.input_channel = input_channel
        self.hm_head = DetHead(hm_list,self.input_channel)
        offset_list = [40,2]
        self.offset_head = DetHead(offset_list,self.input_channel)
        corners_list = [60,20,2]
        self.corner_head = DetHead(corners_list,self.input_channel)

    def forward(self,image):

        img_tensor = torch.tensor(image)
        feats = self.dla_model(img_tensor)
        heat_map = self.hm_head(feats[1])
        offsets = self.offset_head(feats[1])
        corners = self.corner_head(feats[1])

        return (heat_map,offsets,corners)






        

