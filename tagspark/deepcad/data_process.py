"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
Other modification is in the header of the function.
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""
import numpy as np
import os
import tifffile as tiff
import random
import math
from torch.utils.data import Dataset
from skimage import io
import torch
from scipy.signal import convolve
import torch.nn.functional as F
import re
from .buildingblocks import SpatialAttention3D

def random_transform(input, target):
    """
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target : the input and target patch before data augmentation
    Return:
        input, target : the input and target patch after data augmentation
    """
    p_trans = random.randrange(8)
    if p_trans == 0:  # no transformation
        input = input
        target = target
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
        target = np.rot90(target, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
        target = target[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=3, axes=(1, 2))
    return input, target


class trainset(Dataset):
    """
    Train set generator for pytorch training
    
    ---
    Modify the function and the description.
    DeepCAD-Z ver1. 2023  Kai-Chun Jhan
    ---
    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 原本 mean_all 在 __getitem__ 每次都 np.mean(all stacks) 非常慢
        # 最少差異做法：移到 __init__ 先算一次（不改其它邏輯）
        self.mean_all = np.mean(np.array(self.noise_img_all), axis=0)

    def __getitem__(self, index):
        """
        One entire z-stack image serves as the training image and is split into 2 sub-stacks as input and target. 
        Note that this function will no longer support rotating images to expand the dataset in the case of different x-y sizes of images.
        Args:
            index : the index of 3D x-y-z image used for training
        Return:
            input, target : the consecutive frames of the 3D noisy patch serve as the input and target of the network
        """
        p_exc = random.random()  # generate a random number determinate whether swap input and target

        # 保留你原本的 stack_index 取得方式
        stack_index = self.stack_index[index]

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 重大 bug：index 是 patch index (0~num_patches-1)，不是 stack index (0~num_stacks-1)
        # 你原本寫 noise_img = self.noise_img_all[index] 會在 index>=num_stacks 時爆掉
        # 正確應該用 stack_index（每個 patch 對應到哪個 stack）
        noise_img = self.noise_img_all[stack_index]

        D, H, W = noise_img.shape
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # mean_all 改用 __init__ 預先算好的，避免每個 patch 都重算一次
        mean_all = self.mean_all

        # ----------------------------
        # n
        # ----------------------------

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 你原本 pre/post 用的是 patch index 去做 window
        # 但 noise_img_all 的維度是 stack 維度，所以 window 應該以 stack_index 為中心
        pre_index_n = max(stack_index - 4, 0)

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 你原本用 len(self.stack_index)（patch 數量）當上界，會把 slice 拉到很大
        # 正確上界是 len(self.noise_img_all)（stack 數量）
        post_index_n = min(stack_index + 5, len(self.noise_img_all))  # slice end exclusive

        noise_img_stack_n = np.array(self.noise_img_all[pre_index_n:post_index_n])

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 下面 if/elif/else 用到 index±1，本來會在 patch index 最後爆掉
        # 現在改成用 stack_index 判斷邊界，並且索引也用 stack_index±1
        if stack_index == 0:
            noise_img_n = (
                np.mean(noise_img_stack_n, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.5
                + np.array(self.noise_img_all[stack_index + 1]) * 0.05
            )
        elif stack_index == len(self.noise_img_all) - 1:
            noise_img_n = (
                np.mean(noise_img_stack_n, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.5
                + np.array(self.noise_img_all[stack_index - 1]) * 0.05
            )
        else:
            noise_img_n = (
                np.mean(noise_img_stack_n, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.45
                + np.array(self.noise_img_all[stack_index - 1]) * 0.05
                + np.array(self.noise_img_all[stack_index + 1]) * 0.05
            )

        # ----------------------------
        # s
        # ----------------------------

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 同 n：window 以 stack_index 為中心 + 上界用 stack 數量
        pre_index_s = max(stack_index - 4, 0)
        post_index_s = min(stack_index + 5, len(self.noise_img_all))
        noise_img_stack_s = np.array(self.noise_img_all[pre_index_s:post_index_s])

        # ### <<< MODIFIED >>>
        # ### <<< WHY >>>
        # 同 n：邊界判斷與索引改用 stack_index
        if stack_index == 0:
            noise_img_s = (
                np.mean(noise_img_stack_s, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.5
                + np.array(self.noise_img_all[stack_index + 1]) * 0.05
            )
        elif stack_index == len(self.noise_img_all) - 1:
            noise_img_s = (
                np.mean(noise_img_stack_s, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.5
                + np.array(self.noise_img_all[stack_index - 1]) * 0.05
            )
        else:
            noise_img_s = (
                np.mean(noise_img_stack_s, axis=0) * 0.45
                + np.array(self.noise_img_all[stack_index]) * 0.45
                + np.array(self.noise_img_all[stack_index - 1]) * 0.05
                + np.array(self.noise_img_all[stack_index + 1]) * 0.05
            )

        # ----------------------------
        # Noise2Noise: odd/even split
        # ----------------------------
        input_n = noise_img_n[init_s:end_s:2, init_h:end_h, init_w:end_w]
        input_s = noise_img_s[init_s:end_s:2, init_h:end_h, init_w:end_w]
        input_mean = mean_all[init_s:end_s:2, init_h:end_h, init_w:end_w]
        target_n = noise_img_n[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]
        target_s = noise_img_s[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]

        if (end_s % 2) == 1:
            if p_exc < 0.5:
                input_n = input_n[1:, :, :]
                input_s = input_s[1:, :, :]
                input_mean = input_mean[1:, :, :]
            else:
                input_n = input_n[:-1, :, :]
                input_s = input_s[:-1, :, :]
                input_mean = input_mean[:-1, :, :]

        input_n = torch.from_numpy(np.expand_dims(input_n, 0).copy()).float()
        input_s = torch.from_numpy(np.expand_dims(input_s, 0).copy()).float()
        input_mean = torch.from_numpy(np.expand_dims(input_mean, 0).copy()).float()
        target_n = torch.from_numpy(np.expand_dims(target_n, 0).copy()).float()
        target_s = torch.from_numpy(np.expand_dims(target_s, 0).copy()).float()

        return input_n, target_n, input_s, target_s, input_mean

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    """
    Test set generator for pytorch inference

    """

    def __init__(self, name_list, coordinate_list, noise_img_n, noise_img_s):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_n = noise_img_n
        self.noise_img_s = noise_img_s

    def __getitem__(self, index):
        """
        Generate the sub-stacks of the noisy image.
        Args:
            index : the index of 3D patch used for testing
        Return:
            noise_patch : the sub-stacks of the noisy image
            single_coordinate : the specific coordinate of sub-stacks in the noisy image for stitching all sub-stacks
        """
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch_n = self.noise_img_n[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch_s = self.noise_img_s[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch_n = torch.from_numpy(np.expand_dims(noise_patch_n, 0))
        noise_patch_s = torch.from_numpy(np.expand_dims(noise_patch_s, 0))
        return noise_patch_n, noise_patch_s, single_coordinate

    def __len__(self):
        return len(self.name_list)


def get_gap_t(args, img, stack_num):
    whole_x = img.shape[2]
    whole_y = img.shape[1]
    whole_t = img.shape[0]
    print('whole_x -----> ', whole_x)
    print('whole_y -----> ', whole_y)
    print('whole_t -----> ', whole_t)
    w_num = math.floor((whole_x - args.patch_x) / args.gap_x) + 1
    h_num = math.floor((whole_y - args.patch_y) / args.gap_y) + 1
    s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    gap_t = math.floor((whole_t - args.patch_t * 2) / (s_num - 1))
    # print('gap_t -----> ',gap_t)
    return gap_t


def train_preprocess_lessMemoryMulStacks(args):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t * 2
    gap_y = args.gap_y
    gap_x = args.gap_x
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    coordinate_list = {}
    stack_index = []
    noise_im_all = []
    ind = 0;
    print('\033[1;31mImage list for training -----> \033[0m')
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total stack number -----> ', stack_num)

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print(im_name)
        im_dir = im_folder + '//' + im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0] > args.select_img_num:
            noise_im = noise_im[0:args.select_img_num, :, :]
        gap_t2 = get_gap_t(args, noise_im, stack_num)
        args.gap_t = gap_t2
        # print('gap_t2 -----> ',gap_t2)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = noise_im.astype(np.float32) / args.scale_factor  # no preprocessing
        # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor 
        noise_im_all.append(noise_im)

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((whole_y-patch_y+gap_y)/gap_y))
        # print('int((whole_x-patch_x+gap_x)/gap_x) -----> ',int((whole_x-patch_x+gap_x)/gap_x))
        # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((whole_t-patch_t2+gap_t2)/gap_t2))
        for x in range(0, int((whole_y - patch_y + gap_y) / gap_y)):
            for y in range(0, int((whole_x - patch_x + gap_x) / gap_x)):
                for z in range(0, int((whole_t - patch_t2 + gap_t2) / gap_t2)):
                    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder + '_' + im_name.replace('.tif', '') + '_x' + str(x) + '_y' + str(
                        y) + '_z' + str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1;
    return name_list, noise_im_all, coordinate_list, stack_index


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks (if the batch size equal to 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
        raw_image : the noisy sub-stack
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    output_patch = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate, id, output_image, raw_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks. (if the batch size larger than 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
        raw_image : the noisy sub-stack
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id]
    raw_image_id = raw_image[id]
    output_patch = output_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def test_preprocess_lessMemoryNoTail_chooseOne(args, N):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.patch_y
    gap_x = args.patch_x
    gap_t2 = args.patch_t
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)

    im_name = img_list[N]

    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    # print('noise_im shape -----> ',noise_im.shape)
    # print('noise_im max -----> ',noise_im.max())
    # print('noise_im min -----> ',noise_im.min())
    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    noise_im = noise_im.astype(np.float32) / args.scale_factor
    # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((whole_y-patch_y+gap_y)/gap_y))
    # print('int((whole_x-patch_x+gap_x)/gap_x) -----> ',int((whole_x-patch_x+gap_x)/gap_x))
    # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((whole_t-patch_t2+gap_t2)/gap_t2))
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_folder + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list


def test_preprocess_chooseOne(args, index, scale_factor, noise_img_all):
    """
    Choose one original noisy stack and partition it into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.
        

    ---
    The program has been modified to no longer support the slicing and recombination of images. 
    The current training approach involves using the entire volume image as a single unit.
    DeepCAD-Z ver1. 2023  Kai-Chun Jhan
    ---

    Args:
        args : the train object containing input params for partition
        img_id : the id of the test image
    Returns:
        name_list : the coordinates of 3D patch are indexed by the patch name in name_list
        noise_im : the original noisy stacks
        coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack
        im_name : the file name of the noisy stacks

    """

    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    #gap_y = args.gap_y
    #gap_x = args.gap_x
    #gap_t2 = int(args.patch_t * (1 - args.overlap_factor))
    #gap_t2 = args.patch_t
    gap_y = args.patch_y
    gap_x = args.patch_x
    gap_t2 = args.patch_t
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = args.datasets_path
    
    
    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    split_list = [re.split(r'[._]', filename) for filename in img_list]
    sorted_list = sorted(split_list, key=lambda x: int(x[1]))
    sorted_filenames = ['_'.join(parts[:-1]) + '.' + parts[-1] for parts in sorted_list]
    img_list = sorted_filenames
    
    im_name = img_list[index]
    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    input_data_type = noise_im.dtype
    # Minus mean before training
    noise_im = noise_im.astype(np.float32)*args.scale_factor
    img_mean = noise_im.mean()
    noise_im = noise_im-img_mean
    # noise_im_mean = np.mean(noise_img_all,axis=0)

    ### n
    pre_index_n = max(index-4,0)
    post_index_n = min(index+5,noise_img_all.shape[0])
    noise_img_stack_n = np.array(noise_img_all[pre_index_n:post_index_n])
    if index == 0:
        noise_im_n = np.mean(noise_img_stack_n,axis=0)*0.45 + np.array(noise_img_all[index])*0.5 + np.array(noise_img_all[index+1])*0.05
    elif index == int(noise_img_all.shape[0]-1):
        noise_im_n = np.mean(noise_img_stack_n,axis=0)*0.45 + np.array(noise_img_all[index])*0.5 + np.array(noise_img_all[index-1])*0.05
    else:
        noise_im_n = (np.mean(noise_img_stack_n,axis=0)*0.45 + np.array(noise_img_all[index])*0.45 + 
                        np.array(noise_img_all[index-1])*0.05 + np.array(noise_img_all[index+1])*0.05)
    
    ### s
    pre_index_s = max(index-4,0)
    post_index_s = min(index+5,noise_img_all.shape[0])
    noise_img_stack_s = np.array(noise_img_all[pre_index_s:post_index_s])
    if index == 0:
        noise_im_s = np.mean(noise_img_stack_s,axis=0)*0.45 + np.array(noise_img_all[index])*0.5 + np.array(noise_img_all[index+1])*0.05
    elif index == int(noise_img_all.shape[0]-1):
        noise_im_s = np.mean(noise_img_stack_s,axis=0)*0.45 + np.array(noise_img_all[index])*0.5 + np.array(noise_img_all[index-1])*0.05
    else:
        noise_im_s = (np.mean(noise_img_stack_s,axis=0)*0.45 + np.array(noise_img_all[index])*0.45 + 
                        np.array(noise_img_all[index-1])*0.05 + np.array(noise_img_all[index+1])*0.05)

    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    if args.print_img_name:
       print('Testing image name -----> ', im_name)
       print('Testing image shape -----> ', noise_im.shape)

    noise_im_n = torch.from_numpy(noise_im_n).float()
    noise_im_s = torch.from_numpy(noise_im_s).float()
    ########
    
    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_z = noise_im.shape[0]

    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}

    single_coordinate['end_h'] = whole_y
    single_coordinate['end_w'] = whole_x
    single_coordinate['end_s'] = whole_z

    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
    patch_name = args.datasets_name
    # train_raw.append(noise_patch1.transpose(1,2,0))
    name_list.append(patch_name)
    # print(' single_coordinate -----> ',single_coordinate)
    coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im_n, noise_im_s, coordinate_list, im_name, img_mean, input_data_type
