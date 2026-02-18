"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
from .model_3DUnet import UNet3D
import torch.nn as nn
import torch
from .buildingblocks import DoubleConv, SingleConv

class Network_3D_Unet(nn.Module):
    def __init__(self, UNet_type = '3DUNet', in_channels=1, out_channels=1, xy_scale=1, f_maps=64, encode_module=DoubleConv, final_sigmoid = True, conv_kernel=3, pool_type='max'):
        super(Network_3D_Unet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        
        if UNet_type == '3DUNet':
            self.Generator = UNet3D( in_channels = in_channels,
                                     out_channels = out_channels,
                                     xy_scale = xy_scale,
                                     f_maps = f_maps,
                                     layer_order='cr',
                                     encode_module = encode_module,
                                     final_sigmoid = final_sigmoid,
                                     conv_kernel = conv_kernel,
                                     pool_type = pool_type)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x