"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
import importlib
import numpy as np
import torch
import torch.nn as nn

from .buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from .utils import create_feature_maps


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, encode_module=DoubleConv, xy_scale=1, f_maps=64, layer_order='cr', num_groups=8, conv_kernel=3, pool_type='max',
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)
        
        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            # print('out feature num=',out_feature_num)
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=encode_module,
                                  conv_layer_order=layer_order, num_groups=num_groups, conv_kernel_size=conv_kernel)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, pool_type=pool_type,basic_module=encode_module,
                                  conv_layer_order=layer_order, num_groups=num_groups, conv_kernel_size=conv_kernel)
            encoders.append(encoder)
            
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if encode_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            
            elif encode_module == SingleConv:
                in_feature_num = reversed_f_maps[i]
            
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=encode_module, xy_scale=xy_scale,
                              conv_layer_order=layer_order, num_groups=num_groups, kernel_size=conv_kernel)
            decoders.append(decoder)
            
        self.decoders = nn.ModuleList(decoders)
        
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        
        x = self.final_conv(x)
        
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model') #向a模块中导入c.py中的对象
        clazz = getattr(m, class_name) #getattr() 函数用于返回一个对象属性值。
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


