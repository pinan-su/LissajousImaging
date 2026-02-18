"""
This file will help to demonstrate pipeline for testing microscopy data using the DeepCAD-Z algorithm.
The demo shows how to construct the params and call the relevant functions for testing DeepCAD-Z network.
See inside for details.

This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
from deepcad.test_collection_v2 import testing_class
from deepcad.utils import get_first_filename

# %% Select file(s) to be processed (download if not present)
datasets_path = r"C:/Users/user/Desktop/蘇品安/lissajous_v2_preprocess/output/cuda/1/z31/xyzt_volumes"  # folder containing tif files for testing
denoise_model = f'xyzt_volumes_202602181637'  # A folder containing pth models to be tested

# %% First setup some parameters for testing
scale_factor = 1            # intensity scale factor
GPU = '0'
num_workers = 0                       # if you use Windows system, set this to 0.

# %% Setup some parameters for result visualization during testing period (optional)


test_dict = {
    # dataset dependent parameters
    'scale_factor': scale_factor,                   # the factor for image intensity scaling
    'datasets_path': datasets_path,
    'pth_dir': './pth',                 # pth file root path
    'denoise_model' : denoise_model,
    'output_dir' : './results',         # result file root path
    # network related parameters
    'fmap': 16,                          # the number of feature maps
    'encode_module': 'DoubleConv',
    'GPU': GPU,
    'num_workers': num_workers,

    # ===== Fuse 選項 =====
    "fuse_mode": "weighted_avg",  # "weighted_avg"或 "detail_inject"
    "fuse_alpha": 0.5,                 # weighted_avg 用（0~1）: fused = alpha_fuse * N + (1 - alpha_fuse) * S
    "fuse_beta": 0.45,                 # detail_inject: fused = S + beta * (N - S)
    "apply_final_contrast": True,

}

# %%% Testing preparation
# first we create a testing class object with the specified parameters
tc = testing_class(test_dict)
# start the testing process
tc.run()
