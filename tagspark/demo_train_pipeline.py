"""
This file will demonstrate pipeline for training microscopy data using the DeepCAD-Z algorithm.
The demo shows how to construct the params and call the relevant functions for training DeepCAD-Z network.
See inside for details.

This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
from deepcad.train_collection import training_class
from deepcad.utils import get_first_filename

# %% Select file(s) to be processed

datasets_path = r"C:/Users/user/Desktop/蘇品安/lissajous_v2_preprocess/output/cuda/1/z31/xyzt_volumes"  # folder containing tif files for training

# %% First setup some parameters for training
n_epochs = 10               # the number of training epochs
xy_scale = 1
GPU = '0'                   # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
pth_dir = './pth'           # pth file and visualization result file path
num_workers = 0             # if you use Windows system, set this to 0.

# %% Setup some parameters for result visualization during training period (optional)
train_dict = {
    "datasets_path": datasets_path,
    "n_epochs": 10,
    "batch_size": 1,

    "patch_t": 31,
    "patch_x": 128,
    "patch_y": 128,
    "gap_t": 31,
    "gap_x": 100,
    "gap_y": 100,

    "lr": 2.5e-5,
    "fmap": 16,
    "encode_module": "DoubleConv",

   "fuse_beta": 0.55,
   "fuse_weight": 0.15,
   "fuse_beta_jitter": 0.00,
   "show_loss_record": True,
}

# %%% Training preparation
# first we create a training class object with the specified parameters
tc = training_class(train_dict)
# start the training process
#tc.train()
tc.run()