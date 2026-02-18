"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
Removed unnecessary modules like movie_display.
Other modification is in the header of the function.
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
import os
import datetime

import numpy as np
import yaml
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from .network import Network_3D_Unet
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import datetime
from .data_process import trainset, test_preprocess_chooseOne, testset, multibatch_test_save, singlebatch_test_save
from .buildingblocks import DoubleConv, SingleConv
from skimage import io
import re


class training_class():
    """
    Class implementing training process
    """


    def __init__(self, params_dict):
        """
        Constructor class for training process

        Args:
           params_dict: dict
               The collection of training params set by users
        Returns:
           self

        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.n_epochs = 20
        self.xy_scale = 1
        self.fmap = 16
        self.encode_module = SingleConv
        self.output_dir = './results'
        self.pth_dir = './pth'
        self.onnx_dir = './onnx'
        self.batch_size = 1
        self.patch_t = 32
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 32
        self.lr = 0.00001
        self.b1 = 0.5
        self.b2 = 0.999
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.scale_factor = 1
        self.train_datasets_size = 2000
        self.select_img_num = 1000
        self.test_datasize = 400  # how many slices to be tested (use the first image in the folder by default)
        self.visualize_images_per_epoch = False
        self.save_test_images_per_epoch = False
        self.colab_display = False
        self.result_display = ''
        self.show_loss_record = False
        self.set_params(params_dict)
        
    def run(self):
        """
        General function for training DeepCAD network.
        
        ---
        Modify train_preprocess_lessMemoryMulStacks() to train_preprocess()
        DeepCAD-Z ver1. 2023  Kai-Chun Jhan
        ---
        """
        # create some essential file for result storage
        self.prepare_file()
        # crop input tiff file into 3D patches
        self.train_preprocess_lessMemoryMulStacks() # <<< MODIFIED: 使用會切 patch 的版本
        # save some essential training parameters in para.yaml
        self.save_yaml_train()
        # initialize denoise network with training parameters.
        self.initialize_network()
        # specifies the GPU for the training program.
        self.distribute_GPU()
        # start training and result visualization during training period (optional)
        self.train()


    def prepare_file(self):
        """
        Make data folder to store training results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage

        """
        if self.datasets_path[-1]!='/':
           self.datasets_name=self.datasets_path.split("/")[-1]
        else:
            self.datasets_name=self.datasets_path.split("/")[-2]
        pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.pth_path = self.pth_dir + '/' + pth_name
        self.onnx_path = self.onnx_dir + '/' + pth_name
        if not os.path.exists(self.pth_path):
            os.makedirs(self.pth_path)
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def set_params(self, params_dict):
        """
        Set the params set by user to the training class object and calculate some default parameters for training

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        """
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  # patch gap in t """
        self.ngpu = str(self.GPU).count(',') + 1                    # check the number of GPU used for training
        self.batch_size = self.ngpu                                 # By default, the batch size is equal to the number of GPU for minimal memory consumption
        print('\033[1;31mTraining parameters -----> \033[0m')
        print(self.__dict__)


    def initialize_network(self):
        """
        Initialize U-Net 3D network, which is the main network architecture of DeepCAD

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network

        """
        if self.encode_module == 'SingleConv':
            self.encode_module = SingleConv
        elif self.encode_module == 'DoubleConv':
            self.encode_module = DoubleConv
        cuda = True if torch.cuda.is_available() else False
        denoise_generator_n = Network_3D_Unet(in_channels=1,
                                            out_channels=1,
                                            xy_scale=self.xy_scale,
                                            f_maps=self.fmap,
                                            encode_module=self.encode_module,
                                            final_sigmoid=True,
                                            conv_kernel=3,
                                            pool_type='max').cuda()
        denoise_generator_s = Network_3D_Unet(in_channels=1,
                                            out_channels=1,
                                            xy_scale=self.xy_scale,
                                            f_maps=self.fmap,
                                            encode_module=self.encode_module,
                                            final_sigmoid=True,
                                            conv_kernel=3,
                                            pool_type='avg').cuda()
        self.local_model_n = denoise_generator_n
        self.local_model_s = denoise_generator_s

    def get_gap_t(self):
        """
        Calculate the patch gap in t according to the size of input data and the patch gap in x and y

        Important Fields:
           self.gap_t: the patch gap in t.

        """
        w_num = math.floor((self.whole_x - self.patch_x) / self.gap_x) + 1
        h_num = math.floor((self.whole_y - self.patch_y) / self.gap_y) + 1
        s_num = math.ceil(self.train_datasets_size / w_num / h_num / self.stack_num)
        self.gap_t = math.floor((self.whole_t - self.patch_t * 2) / (s_num - 1))
    
       
    def train_preprocess_lessMemoryMulStacks(self):
        """
        The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
        overlap factor in each dimension.

        Important Fields:
           self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
           self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
           self.stack_index : the index of the noisy stacks.
           self.noise_im_all : the collection of all noisy stacks.

        """
        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.noise_im_all = []
        ind = 0

        print('\033[1;31mImage list for training -----> \033[0m')
        file_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
        self.stack_num = len(file_list)
        print('Total stack number -----> ', self.stack_num)

        for im_name in file_list:
            print('Noise image name -----> ', im_name)
            im_dir = self.datasets_path + '//' + im_name
            noise_im = tiff.imread(im_dir)

            if noise_im.shape[0] > self.select_img_num:
                noise_im = noise_im[0:self.select_img_num, :, :]

            self.whole_x = noise_im.shape[2]
            self.whole_y = noise_im.shape[1]
            self.whole_t = noise_im.shape[0]
            print('Noise image shape -----> ', noise_im.shape)

        # ==============================
        # <<< MODIFIED >>>
        # 不再把 gap 強制覆蓋成 patch
        # （改成使用 set_params() 計算好的 gap_x/gap_y/gap_t）
        # ==============================

            # --- 建議：gap 用你參數的設定（不要硬改成 patch）---
            # 如果你想用 overlap_factor，自動算 gap，打開下面三行，並註解掉後面 gap_* 的 max(1, ...)：
            # self.gap_x = max(1, int(self.patch_x * (1 - self.overlap_factor)))
            # self.gap_y = max(1, int(self.patch_y * (1 - self.overlap_factor)))
            # self.gap_t = max(1, int(self.patch_t * (1 - self.overlap_factor)))

            # 確保 gap 不為 0
            self.gap_x = max(1, int(self.gap_x))
            self.gap_y = max(1, int(self.gap_y))
            self.gap_t = max(1, int(self.gap_t))

            # ==============================
            # <<< MODIFIED >>>
            # patch 不能大於 whole（否則永遠切不到）
            # 例如 whole_t=31, patch_t=32 => patch_t clamp 成 31
            # ==============================
            # --- 關鍵：patch_t 不可以大於 whole_t，否則永遠切不到 ---
            patch_x = min(self.patch_x, self.whole_x)
            patch_y = min(self.patch_y, self.whole_y)
            patch_t = min(self.patch_t, self.whole_t)

            # Minus mean before training
            noise_im = noise_im.astype(np.float32) * self.scale_factor
            noise_im = noise_im - noise_im.mean()
            self.noise_im_all.append(noise_im)

            # ==============================
            # <<< MODIFIED >>>
            # 修正 patch 數量計算方式：
            # 用 floor((whole - patch)/gap)+1，避免 int(whole/gap) 造成 range=0
            # ==============================
            # 正確計算「能放幾個 patch」：至少 1 個（若 whole >= patch）
            n_y = max(1, math.floor((self.whole_y - patch_y) / self.gap_y) + 1) if self.whole_y >= patch_y else 1
            n_x = max(1, math.floor((self.whole_x - patch_x) / self.gap_x) + 1) if self.whole_x >= patch_x else 1
            n_t = max(1, math.floor((self.whole_t - patch_t) / self.gap_t) + 1) if self.whole_t >= patch_t else 1

            for x in range(n_y):
                for y in range(n_x):
                    for z in range(n_t):
                        init_h = x * self.gap_y
                        init_w = y * self.gap_x
                        init_s = z * self.gap_t

                        end_h = init_h + patch_y
                        end_w = init_w + patch_x
                        end_s = init_s + patch_t

                        # ==============================
                        # <<< MODIFIED >>>
                        # 超界就貼齊：確保 end 不會 > whole
                        # ==============================
                        # 超界就往回貼齊（保證 end 不超過 whole）
                        if end_h > self.whole_y:
                            end_h = self.whole_y
                            init_h = max(0, end_h - patch_y)

                        if end_w > self.whole_x:
                            end_w = self.whole_x
                            init_w = max(0, end_w - patch_x)

                        if end_s > self.whole_t:
                            end_s = self.whole_t
                            init_s = max(0, end_s - patch_t)

                        single_coordinate = {
                            'init_h': int(init_h), 'end_h': int(end_h),
                            'init_w': int(init_w), 'end_w': int(end_w),
                            'init_s': int(init_s), 'end_s': int(end_s)
                        }

                        patch_name = (
                            self.datasets_name + '_' +
                            im_name.replace('.tiff', '').replace('.tif', '') +
                            f'_x{x}_y{y}_z{z}'
                        )

                        self.name_list.append(patch_name)
                        self.coordinate_list[patch_name] = single_coordinate
                        self.stack_index.append(ind)

            ind += 1
        # ==============================
        # <<< MODIFIED >>>
        # 加 debug：確認真的切出 patch
        # ==============================
        print("[DBG] total patches:", len(self.name_list))
        if len(self.name_list) > 0:
            k = self.name_list[0]
            print("[DBG] first coord:", self.coordinate_list[k])

    def train_preprocess(self):
      """
        Preprocess function for training that can parse data name from defined directory and handle the entire Z-stack image without slicing.

        Important Args:
            self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
            self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
            self.stack_index : the index of the noisy stacks.
            self.noise_im_all : the collection of all noisy stacks.

        Modification Details:
            The original function train_preprocess_lessMemoryMulStacks() has been modified to become train_preprocess(). 
            This modification allows the function to process the entire Z-stack image without the need for slicing. 
            Also named the third-dimestion as "z".

        DeepCAD-Z ver1. 2023  Kai-Chun Jhan
      """
      self.name_list = []
      self.coordinate_list = {}
      self.stack_index = []
      self.noise_im_all = []
      ind = 0
      print('\033[1;31mImage list for training -----> \033[0m')
      self.stack_num = len(list(os.walk(self.datasets_path, topdown=False))[-1][-1])
      print('Total stack number -----> ', self.stack_num)
      
      file_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
      if len(file_list)>1:
          split_list = [re.split(r'[._]', filename) for filename in file_list]
          sorted_list = sorted(split_list, key=lambda x: int(x[1]))
          sorted_filenames = ['_'.join(parts[:-1]) + '.' + parts[-1] for parts in sorted_list]
      else:
          sorted_filenames = file_list
      # print(sorted_filenames)
      for num in range(len(sorted_filenames)):
          im_name = sorted_filenames[num]
          print('Noise image name -----> ', im_name)
          im_dir = self.datasets_path + '//' + im_name
          noise_im = tiff.imread(im_dir)
          noise_im = noise_im
          if noise_im.shape[0] > self.select_img_num:
              noise_im = noise_im[0:self.select_img_num, :, :]
          self.whole_x = noise_im.shape[2]
          self.whole_y = noise_im.shape[1]
          self.whole_z = noise_im.shape[0]
          print('Noise image shape -----> ', noise_im.shape)
          
          # Calculate real gap_t
          # self.get_gap_t()

          # No preprocessing
          # noise_im = noise_im.astype(np.float32) / self.scale_factor
          # Minus mean before training
          input_data_type = noise_im.dtype
        
          noise_im = noise_im.astype(np.float32)*self.scale_factor
          noise_im = noise_im-noise_im.mean()

          self.noise_im_all.append(noise_im)
          single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
          single_coordinate['end_h'] = self.whole_y
          single_coordinate['end_w'] = self.whole_x
          single_coordinate['end_s'] = self.whole_z
          patch_name = self.datasets_name
          self.name_list.append(patch_name)
          self.coordinate_list[patch_name] = single_coordinate
          self.stack_index.append(ind)
              
          ind = ind + 1

    def save_yaml_train(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.pth_path + '//para.yaml'
        para = {'n_epochs': 0, 'datasets_path': 0, 'overlap_factor': 0,
                'output_dir': 0, 'pth_path': 0, 'encode_module': 0,
                'GPU': 0, 'batch_size': 0, 'xy_scale': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'lr': 0, 'b1': 0, 'b2': 0, 'fmap': 0, 'scale_factor': 0,
                'select_img_num': 0, 'train_datasets_size': 0}
        para["n_epochs"] = self.n_epochs
        para["datasets_path"] = self.datasets_path
        para["output_dir"] = self.output_dir
        para["pth_path"] = self.pth_path
        para["encode_module"] = self.encode_module
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["xy_scale"] = self.xy_scale
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["lr"] = self.lr
        para["b1"] = self.b1
        para["b2"] = self.b2
        para["fmap"] = self.fmap
        para["scale_factor"] = self.scale_factor
        para["select_img_num"] = self.select_img_num
        para["train_datasets_size"] = self.train_datasets_size
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the training program. Print the using GPU information to the screen.
        For acceleration, multiple GPUs parallel training is recommended.

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model_n = self.local_model_n.cuda()
            self.local_model_n = nn.DataParallel(self.local_model_n, device_ids=range(self.ngpu))
            self.local_model_s = self.local_model_s.cuda()
            self.local_model_s = nn.DataParallel(self.local_model_s, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(self.ngpu))

    def train(self):
        """
        Pytorch training workflow (memory-optimized)
        - Train N model then S model sequentially to avoid holding two graphs at once
        - Use AMP mixed precision to reduce GPU memory usage
        """
        # ==============================
        # <<< MODIFIED >>>
        # 加入 debug helper：用來印出 tensor min/max/mean/std
        # 用於判斷 loss=0 是印錯 or 資料問題
        #加入loss helper
        # ==============================
        def grad_loss_3d(pred, target):
            # pred/target: (B,1,T,H,W)
            dz_p = pred[:,:,1:]-pred[:,:,:-1]
            dy_p = pred[:,:,:,1:]-pred[:,:,:,:-1]
            dx_p = pred[:,:,:,:,1:]-pred[:,:,:,:,:-1]

            dz_t = target[:,:,1:]-target[:,:,:-1]
            dy_t = target[:,:,:,1:]-target[:,:,:,:-1]
            dx_t = target[:,:,:,:,1:]-target[:,:,:,:,:-1]

            return (dz_p-dz_t).abs().mean() + (dy_p-dy_t).abs().mean() + (dx_p-dx_t).abs().mean()

        def tv_loss_3d(x):
            dz = (x[:,:,1:]-x[:,:,:-1]).abs().mean()
            dy = (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
            dx = (x[:,:,:,:,1:]-x[:,:,:,:,:-1]).abs().mean()
            return dz + dy + dx

        def _dbg_tensor(name, t):
            # t: torch.Tensor on GPU/CPU
            t_det = t.detach()
            info = {
                "shape": tuple(t_det.shape),
                "dtype": str(t_det.dtype),
                "device": str(t_det.device),
                "min": float(t_det.min().item()),
                "max": float(t_det.max().item()),
                "mean": float(t_det.mean().item()),
                "std": float(t_det.std(unbiased=False).item()),
                "zero_ratio": float((t_det == 0).float().mean().item()),
                "nan": bool(torch.isnan(t_det).any().item()),
                "inf": bool(torch.isinf(t_det).any().item()),
            }
            print(f"[DBG] {name}: {info}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA is not available. DeepCAD-RT training expects a GPU here.")

        # Optimizers
        optimizer_N = torch.optim.Adam(self.local_model_n.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_S = torch.optim.Adam(self.local_model_s.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Loss
        L1_pixelwise = torch.nn.L1Loss().to(device)
        L2_pixelwise = torch.nn.MSELoss().to(device)

        # ==============================
        # <<< MODIFIED >>>
        # AMP 混合精度：省顯存
        # ==============================
        # AMP
        use_amp = False
        scaler_N = torch.cuda.amp.GradScaler(enabled=use_amp)
        scaler_S = torch.cuda.amp.GradScaler(enabled=use_amp)

        prev_time = time.time()
        time_start = time.time()

        loss_rec = []
        mae_rec = []
        mse_rec = []

        # Debug only (VERY slow + more memory). Keep False for normal training.
        # torch.autograd.set_detect_anomaly(True)

        torch.cuda.reset_peak_memory_stats()

        for epoch in range(0, self.n_epochs):
            train_data = trainset(self.name_list, self.coordinate_list, self.noise_im_all, self.stack_index)

            trainloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True  # helps host->GPU copy
            )

            loss_list = []
            mae_list = []
            mse_list = []

            # Set train mode
            self.local_model_n.train()
            self.local_model_s.train()

            for iteration, (input_n, target_n, input_s, target_s, input_mean) in enumerate(trainloader):
                # ---- move to GPU ----
                # ==============================
                # <<< MODIFIED >>>
                # N 和 S 分開 forward/backward/step
                # 避免兩個 graph 同時佔用顯存
                # ==============================
                input_n = input_n.to(device, non_blocking=True)
                target_n = target_n.to(device, non_blocking=True)

                input_s = input_s.to(device, non_blocking=True)
                target_s = target_s.to(device, non_blocking=True)      # (kept if you need it later)
                input_mean = input_mean.to(device, non_blocking=True)

                if epoch == 0 and iteration == 0:
                    print("\n========== DEBUG FIRST BATCH ==========")
                    _dbg_tensor("input_n", input_n)
                    _dbg_tensor("target_n", target_n)
                    _dbg_tensor("input_s", input_s)
                    _dbg_tensor("input_mean", input_mean)

                    # 直接看 input/target 本來就有多像（如果超像，loss 會接近 0）
                    mae_n0 = torch.mean(torch.abs(input_n - target_n)).item()
                    mae_s0 = torch.mean(torch.abs(input_s - input_mean)).item()
                    print(f"[DBG] MAE(input_n, target_n)  = {mae_n0:.6e}")
                    print(f"[DBG] MAE(input_s, input_mean)= {mae_s0:.6e}")

                    # 顯示你目前 print 格式是否把小數顯示成 0
                    print("[DBG] If losses are < 1e-4, '%.4f' will show 0.0000")
                    print("======================================\n")
                    
                # =========================
                # (1) Train N model
                # =========================
                optimizer_N.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_B_n = self.local_model_n(input_n)
                    L1_loss_n = L1_pixelwise(fake_B_n, target_n)
                    L2_loss_n = L2_pixelwise(fake_B_n, target_n)
                    #Total_loss_n = L1_loss_n + torch.sqrt(L2_loss_n + 1e-12)

                    # N: 重視結構（保細節）
                    w_g_n = 0.06   # 0.03~0.10
                    Total_loss_n = L1_loss_n + 0.5*torch.sqrt(L2_loss_n + 1e-12) + w_g_n * grad_loss_3d(fake_B_n, target_n)


                scaler_N.scale(Total_loss_n).backward()
                scaler_N.step(optimizer_N)
                scaler_N.update()

                # Free N graph tensors ASAP
                del fake_B_n

                # =========================
                # (2) Train S model
                # =========================
                optimizer_S.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_B_s = self.local_model_s(input_s)
                    L1_loss_s = L1_pixelwise(fake_B_s, input_mean)
                    L2_loss_s = L2_pixelwise(fake_B_s, input_mean)
                    #Total_loss_s = L1_loss_s + torch.sqrt(L2_loss_s + 1e-12)

                    # S: 更乾淨（底圖）
                    w_g_s = 0.03   # 比 N 小
                    w_tv  = 0.01   # 0.005~0.02
                    Total_loss_s = L1_loss_s + 0.5*torch.sqrt(L2_loss_s + 1e-12) + w_g_s * grad_loss_3d(fake_B_s, input_mean) + w_tv * tv_loss_3d(fake_B_s)


                scaler_S.scale(Total_loss_s).backward()
                scaler_S.step(optimizer_S)
                scaler_S.update()

                del fake_B_s

                # ---- logging (detach to avoid graph) ----
                total_loss_val = (Total_loss_n.detach() + Total_loss_s.detach()).item()
                mae_val = (L1_loss_n.detach() + L1_loss_s.detach()).item()
                mse_val = (L2_loss_n.detach() + L2_loss_s.detach()).item()

                loss_list.append(total_loss_val)
                mae_list.append(mae_val)
                mse_list.append(mse_val)

                # ==============================
                # <<< MODIFIED >>>
                # 第一個 batch 印出 debug（你後來貼的那些）
                # 用來確認真的在訓練、loss 真實數值
                # ==============================
                if epoch == 0 and iteration == 0:
                    print("[DBG] L1_loss_n =", float(L1_loss_n.detach().cpu()))
                    print("[DBG] L2_loss_n =", float(L2_loss_n.detach().cpu()))
                    print("[DBG] Total_loss_n =", float(Total_loss_n.detach().cpu()))
                    print("[DBG] L1_loss_s =", float(L1_loss_s.detach().cpu()))
                    print("[DBG] L2_loss_s =", float(L2_loss_s.detach().cpu()))
                    print("[DBG] Total_loss_s =", float(Total_loss_s.detach().cpu()))
                    
                # ---- ETA ----
                batches_done = epoch * len(trainloader) + iteration
                batches_left = self.n_epochs * len(trainloader) - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()

                if iteration % 1 == 0:
                    time_end = time.time()
                    print(
                        '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.4f, L1 Loss: %.4f, L2 Loss: %.4f] [ETA: %s] [Time cost: %.0f s]     '
                        % (
                            epoch + 1,
                            self.n_epochs,
                            iteration + 1,
                            len(trainloader),
                            total_loss_val,
                            mae_val,
                            mse_val,
                            time_left,
                            time_end - time_start
                        ),
                        end=' '
                    )

                # ---- end of epoch ----
                if (iteration + 1) % len(trainloader) == 0:
                    print('\n', end=' ')
                    self.save_model(epoch, iteration)

                    if (self.visualize_images_per_epoch or self.save_test_images_per_epoch):
                        print(f'Testing model of epoch {epoch + 1} on the first noisy file ----->')
                        # IMPORTANT: test() should use torch.no_grad() internally
                        self.test(epoch, iteration)
                        print('\n', end=' ')

                # Explicitly delete batch tensors to reduce peak memory
                del input_n, target_n, input_s, target_s, input_mean
                del Total_loss_n, Total_loss_s, L1_loss_n, L1_loss_s, L2_loss_n, L2_loss_s

            loss_rec.append(float(np.mean(loss_list)) if len(loss_list) else 0.0)
            mae_rec.append(float(np.mean(mae_list)) if len(mae_list) else 0.0)
            mse_rec.append(float(np.mean(mse_list)) if len(mse_list) else 0.0)

        print('Train finished. Save all models to disk.')
        print("max allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")
        print("max reserved :", torch.cuda.max_memory_reserved() / 1024**3, "GB")
    
        if self.colab_display:
            result_img_list = []
            results_path = self.pth_path
            results_list = list(os.walk(results_path, topdown=False))[-1][-1]
            for i in range(len(results_list)):
              aaa = results_list[i]
              if '.tif' in aaa:
                 result_img_list.append(aaa)
            result_img_list.sort()
            self.result_display = results_path+'/'+result_img_list[-1]
            
        if self.show_loss_record:
            plt.figure()
            plt.plot(np.arange(1,self.n_epochs+1), loss_rec)
            plt.legend(['loss'])
            plt.xlabel('epoch')
            plt.ylabel('')
            plt.savefig(self.pth_path+'/'+'loss.png')
            plt.figure()
            plt.plot(np.arange(1,self.n_epochs+1), mae_rec)
            plt.legend(['mae'])
            plt.xlabel('epoch')
            plt.ylabel('')
            plt.savefig(self.pth_path+'/'+'mae.png')
            plt.figure()
            plt.plot(np.arange(1,self.n_epochs+1), mse_rec)
            plt.legend(['mse'])
            plt.xlabel('epoch')
            plt.ylabel('')
            plt.savefig(self.pth_path+'/'+'mse.png')

    def save_model(self, epoch, iteration):
        """
        Model storage.
        Args:
           train_epoch : current train epoch number
           train_iteration : current train_iteration number
        """
        model_save_name_n = self.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(4) + '_n.pth'
        model_save_name_s = self.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(4) + '_s.pth'
        if isinstance(self.local_model_n, nn.DataParallel):
            torch.save(self.local_model_n.module.state_dict(), model_save_name_n)  # parallel
            torch.save(self.local_model_s.module.state_dict(), model_save_name_s)  # parallel
        else:
            torch.save(self.local_model_n.state_dict(), model_save_name_n)  # not parallel
            torch.save(self.local_model_s.state_dict(), model_save_name_s)  # not parallel
        # covert pth to onnx
        onnx_save_name_n = self.onnx_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(4
        ) + '_Patch_'+ str(self.patch_x) + '_' + str(self.patch_y) + '_' + str(self.patch_t) + '_n.onnx'
        onnx_save_name_s = self.onnx_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(4
        ) + '_Patch_'+ str(self.patch_x) + '_' + str(self.patch_y) + '_' + str(self.patch_t) + '_s.onnx'
        input_name = ['input']
        output_name = ['output']

        input = torch.randn(1, 1, self.patch_t, self.patch_x,  self.patch_y, requires_grad=True).cuda()
        torch.onnx.export(self.local_model_n.module, input, onnx_save_name_n, export_params=True,input_names=input_name, output_names=output_name,opset_version=11,verbose=False)
        torch.onnx.export(self.local_model_s.module, input, onnx_save_name_s, export_params=True,input_names=input_name, output_names=output_name,opset_version=11,verbose=False)

    def test(self, train_epoch, train_iteration):
        """
        Pytorch testing workflow
        Args:
            train_epoch : current train epoch number
            train_iteration : current train_iteration number
        """
        # Crop test file into 3D patches for inference
        self.print_img_name = True
        name_list, noise_img, coordinate_list, test_im_name, img_mean, input_data_type = test_preprocess_chooseOne(self, img_id=0)
        # Record the inference time
        prev_time = time.time()
        time_start = time.time()
        denoise_img = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)
        test_data = testset(name_list, coordinate_list, noise_img)
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
            print('test now')
            # Pre-trained models are loaded into memory and the sub-stacks are directly fed into the model.
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = self.local_model(real_A)

            # Determine approximate time left
            batches_done = iteration
            batches_left = 1 * len(testloader) - batches_done
            time_left_seconds = int(batches_left * (time.time() - prev_time))
            time_left = datetime.timedelta(seconds=time_left_seconds)
            prev_time = time.time()
            if iteration % 1 == 0:
                time_end = time.time()
                time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                print(
                    '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                    % (
                        iteration + 1,
                        len(testloader),
                        time_cost,
                        time_left_seconds
                    ), end=' ')

            if (iteration + 1) % len(testloader) == 0:
                print('\n', end=' ')

            # Enhanced sub-stacks are sequentially output from the network
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            if (output_image.ndim == 3):
                postprocess_turn = 1
            else:
                postprocess_turn = output_image.shape[0]

            # The final enhanced stack can be obtained by stitching all sub-stacks.
            if (postprocess_turn > 1):
                for id in range(postprocess_turn):
                    output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                        single_coordinate, id, output_image, raw_image)

                    output_patch=output_patch+img_mean
                    raw_patch=raw_patch+img_mean

                    """ denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5 """
                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / (np.sum(output_patch) + 1e-8))

                    input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = raw_patch
            else:
                output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                    single_coordinate, output_image, raw_image)

                output_patch=output_patch+img_mean
                raw_patch=raw_patch+img_mean

                """ denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5 """
                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / (np.sum(output_patch) + 1e-8))
                input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = raw_patch

        # Stitching finish
        output_img = denoise_img.squeeze().astype(np.float32) * self.scale_factor
        del denoise_img


        # Save inference image
        if (self.save_test_images_per_epoch):
            output_img = output_img[50:self.test_datasize-50, :, :]
            if input_data_type == 'uint16':
                output_img=np.clip(output_img, 0, 65535)
                output_img = output_img.astype('uint16')

            elif input_data_type == 'int16':
                output_img=np.clip(output_img, -32767, 32767)
                output_img = output_img.astype('int16')

            else:
                output_img = output_img.astype('int32')


            result_name = self.pth_path + '//' + test_im_name.replace('.tif', '') + '_' + 'E_' + str(
                train_epoch + 1).zfill(2) + '_Iter_' + str(train_iteration + 1).zfill(4) + '.tif'
            io.imsave(result_name, output_img, check_contrast=False)

