"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='ce', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=int((kernel_size-1)/2)):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=5, order='cr', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels // 1, kernel_size=1)  # 可能使用更少的通道
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()

        if D > 2:
            x_downsampled = F.avg_pool3d(x, kernel_size=2, stride=2)  # 下采样
            D_scale = 2
        else:
            x_downsampled = F.avg_pool2d(x.squeeze(0), kernel_size=2, stride=2)  # 下采样
            x_downsampled = x_downsampled.unsqueeze(0)
            D_scale = 1
        query = self.query_conv(x_downsampled).view(batch_size, -1, (D // D_scale) * (H // 2) * (W // 2))
        key = self.key_conv(x_downsampled).view(batch_size, -1, (D // D_scale) * (H // 2) * (W // 2))
        value = self.value_conv(x_downsampled).view(batch_size, -1, (D // D_scale) * (H // 2) * (W // 2))
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C // 1, D // D_scale, H // 2, W // 2)  # 保持维度
        out = F.interpolate(out, size=(D,H,W), mode='trilinear', align_corners=False)
        return out


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=5, padding = 2):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        avg_out = torch.mean(x, dim=1, keepdim=True)            # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)          # [B, 1, D, H, W]
        attention = torch.cat([avg_out, max_out], dim=1)        # [B, 2, D, H, W]
        attention = self.conv(attention)                        # [B, 1, D, H, W]
        attention = self.sigmoid(attention)                     # [B, 1, D, H, W]
        return x * attention 


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ExtResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=5, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr',
                 num_groups=8):
        super(Encoder, self).__init__()
        
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            elif pool_type == 'avg':
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        
        if basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)
            
        elif basic_module == SingleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)
        
        self.selfA = SelfAttention(out_channels)
        self.spatailA = SpatialAttention3D()
        
    def forward(self, x):
        x = self.spatailA(x)
        
        if self.pooling is not None:
            x = self.pooling(x)
        
        x = self.basic_module(x)
        
        if np.array(x.shape[2]) < 9:
            x = self.selfA(x)
        else:
            x = self.spatailA(x)

        return x
    

class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `ExtDoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(4, 4, 4), basic_module=DoubleConv, xy_scale=1, conv_layer_order='cr', num_groups=8):
        super(Decoder, self).__init__()
        
        self.xy_scale = xy_scale
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
                                  
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels,
                                               kernel_size=3, stride=2,
                                               padding=(0,1,1), output_padding=(0,1,1))
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels
        
        if basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)
         
        elif basic_module == SingleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)
            
    def forward(self, encoder_features, x):
        
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimensio
            x = torch.cat((encoder_features, x), dim=1)

        else:
            # use ConvTranspose3d and summation joining
            x = self.conv3d(x)  # Output shape will be (1, 64, 3, 16, 16)
            D, H, W = encoder_features.size()[2:]
            output_size = [D*1,H*self.xy_scale,W*self.xy_scale]
            x = F.interpolate(x, size=output_size, mode='nearest')
            encoder_features = F.interpolate(encoder_features, size=output_size, mode='nearest')
            x += encoder_features
            
        x = self.basic_module(x)
        return x


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, order='cr', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


