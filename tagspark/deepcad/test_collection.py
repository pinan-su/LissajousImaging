# deepcad/test_collection_v2.py
# ============================================================
# DeepCAD-Z Test Pipeline (v2) - aligned with train_collection_v2.py
#
# 你目前的 train_collection_v2.py 是「直接用模型輸出去對齊目標」：
#   fake_B_n = model_n(input_n)   # detail branch
#   fake_B_s = model_s(input_s)   # smooth/base branch
#
# 因此 Test 預設也採用同樣的輸出語意：
#   pred_n = model_n(x_n)   (不做 tanh/clip/能量校正)
#   pred_s = model_s(x_s)
#
# Fusion（和訓練一致，預設）：
#   fused = S + beta * (N - S)    # detail_inject  (== train 的 fused 定義)
#
# 輸入資料域（和訓練一致）：
#   讀取 volume -> float32 -> 減去該 volume 的 mean（mean-centering）
#   模型輸出也在 mean-centered domain。
#   存檔時會把 mean 加回去，並依原始 dtype 做 clip/cast。
#
# 另外保留選項：
#   - fuse_mode='weighted_avg'：fused = alpha*N + (1-alpha)*S
#   - pipeline_mode='consistent_corr'：舊版「x + alpha*tanh(out) 再 clip 到[0,1]」
#     （⚠️這和 train_collection_v2 的訓練語意不同，僅作為對照用）
# ============================================================
"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
Removed unnecessary modules like movie_display.
Other modification is in the header of the function.
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""

"""
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)
Removed unnecessary modules like movie_display.
Other modification is in the header of the function.
DeepCAD-Z ver1. 2023  Kai-Chun Jhan
"""

import os
import numpy as np
import yaml
from .network import Network_3D_Unet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tifffile as tiff
import time
import datetime
from .data_process import test_preprocess_chooseOne, testset, multibatch_test_save, singlebatch_test_save
from .buildingblocks import DoubleConv, SingleConv
from skimage import io
from scipy.ndimage import zoom
import re
import cv2

# =========================================================
# <<< MODIFIED >>>  (NEW) Weight-map stitching helpers
# <<< WHY >>>       原本 test() 沒有真的 stitch 回整張，只保留最後一塊 patch
#                  你要求用 weight map（重疊區域做加權平均，消除接縫）
# =========================================================

def _hann_1d(n: int) -> np.ndarray:
    """Hann window in [0,1], length n. (smooth edges)"""
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = np.hanning(n).astype(np.float32)
    # 防止端點為 0 造成某些邊界全為 0（尤其 patch 很小）
    eps = np.float32(1e-3)
    w = np.clip(w, eps, 1.0)
    return w

def make_weight_map_3d(t: int, h: int, w: int) -> np.ndarray:
    """
    Create 3D separable weight map: wt(t)*wh(h)*ww(w).
    Smoothly down-weight patch edges -> reduces seams when overlapping patches.
    """
    wt = _hann_1d(t)[:, None, None]
    wh = _hann_1d(h)[None, :, None]
    ww = _hann_1d(w)[None, None, :]
    wm = wt * wh * ww
    return wm.astype(np.float32)

def _coords_from_dataloader_batch(single_coordinate):
    """
    DataLoader 會把 dict 裡的 int 變成 tensor 並且帶 batch 維度：
      single_coordinate['init_h'] -> tensor([.., ..])
    這裡把它轉成 list[dict]，每個 batch element 一個 dict(int)
    """
    # 如果 testset 回傳本來就是 dict（少見），直接包成 batch=1
    if isinstance(single_coordinate, dict):
        # 可能是 dict of tensors，也可能是 dict of ints
        any_v = next(iter(single_coordinate.values()))
        if torch.is_tensor(any_v) and any_v.ndim >= 1:
            B = int(any_v.shape[0])
            coords = []
            for b in range(B):
                coords.append({k: int(single_coordinate[k][b].item()) for k in single_coordinate})
            return coords
        else:
            return [{k: int(single_coordinate[k]) for k in single_coordinate}]

    # 一般情況：DataLoader 給你 dict-like batch（實際型態常是 dict）
    try:
        any_v = next(iter(single_coordinate.values()))
        B = int(any_v.shape[0]) if torch.is_tensor(any_v) else 1
        coords = []
        for b in range(B):
            coords.append({k: int(single_coordinate[k][b].item()) for k in single_coordinate})
        return coords
    except Exception:
        # 保底：當作 batch=1
        return [single_coordinate]


class testing_class():
    """
    Class implementing testing process
    """

    def __init__(self, params_dict):
        """
        Constructor class for testing process

        Args:
           params_dict: dict
               The collection of testing params set by users
        Returns:
           self
        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.fmap = 16
        self.encode_module = SingleConv
        self.output_dir = './results'
        self.pth_dir = ''
        self.batch_size = 1
        self.patch_t = 31
        self.patch_x = 128
        self.patch_y = 128
        self.gap_y = 100
        self.gap_x = 100
        self.gap_t = 16
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.scale_factor = 1
        self.test_datasize = 400
        self.denoise_model = ''
        self.visualize_images_per_epoch = False
        self.save_test_images_per_epoch = True
        self.colab_display = False
        self.result_display = ''
        # =========================================================
        # <<< NEW >>> Fusion settings (must match training_collection_v2)
        # <<< WHY >>> train_collection_v2 的融合目標是：
        #            fused = S + beta*(N - S)
        #            其中 S = local_model_s 輸出（較平滑/穩定），N = local_model_n 輸出（較保細節）
        #            所以 test 端也必須用同一條公式，才能做到「train/test 分佈一致」
        # =========================================================
        self.fuse_mode = 'detail_inject'   # 'detail_inject' or 'weighted_avg'
        self.fuse_beta = 0.55             # detail_inject: fused = S + beta*(N - S)
        self.fuse_alpha = 0.50            # weighted_avg: fused = alpha*N + (1-alpha)*S

        # =========================================================
        # <<< NEW >>> Consistent correction pipeline (default)
        # <<< WHY >>> 避免 legacy energy scaling 造成負值開根號 -> NaN
        #            並且讓 test 與你在註解中定義的「x_in + alpha*tanh(out)」一致
        #
        # pipeline_mode:
        #   - 'consistent'    : pred = clip(x_in + alpha*tanh(out), 0, 1)  (default)
        #   - 'legacy_energy' : 使用舊版「加回 mean + 能量比例校正」(保底模式)
        # =========================================================
        self.pipeline_mode = 'train_v2'

        # dynamic alpha (consistent mode)
        self.alpha_min = 0.05
        self.alpha_max = 0.35
        self.alpha_k   = 0.25

        self.set_params(params_dict)

        # =========================================================
        # <<< MODIFIED >>> (NEW) cache for weight maps by shape
        # <<< WHY >>>      weight map 每次重算很浪費，cache 起來加速
        # =========================================================
        self._weight_cache = {}

    def save_txt_params(self):
        """
        Save testing parameters to a readable text file.
        This helps reproducibility and experiment tracking.
        """

        txt_path = os.path.join(self.output_path, "params.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=============================================\n")
            f.write("DeepCAD-Z Test Parameters Record\n")
            f.write("=============================================\n\n")

            f.write(f"Time: {datetime.datetime.now()}\n\n")

            f.write("=== Dataset ===\n")
            f.write(f"datasets_path: {self.datasets_path}\n")
            f.write(f"test_datasize: {self.test_datasize}\n")
            f.write(f"scale_factor: {self.scale_factor}\n\n")

            f.write("=== Patch Settings ===\n")
            f.write(f"patch_t: {self.patch_t}\n")
            f.write(f"patch_x: {self.patch_x}\n")
            f.write(f"patch_y: {self.patch_y}\n")
            f.write(f"gap_t: {self.gap_t}\n")
            f.write(f"gap_x: {self.gap_x}\n")
            f.write(f"gap_y: {self.gap_y}\n\n")

            f.write("=== Model ===\n")
            f.write(f"fmap: {self.fmap}\n")
            f.write(f"encode_module: {self.encode_module}\n")
            f.write(f"denoise_model: {self.denoise_model}\n\n")

            f.write("=== Dynamic Alpha ===\n")
            f.write(f"alpha_k: {getattr(self, 'alpha_k', 'N/A')}\n")
            f.write(f"alpha_min: {getattr(self, 'alpha_min', 'N/A')}\n")
            f.write(f"alpha_max: {getattr(self, 'alpha_max', 'N/A')}\n\n")

            f.write("=== Fuse Mode ===\n")
            f.write(f"fuse_mode: {getattr(self, 'fuse_mode', 'detail_inject')}\n")
            f.write(f"beta (detail_inject): {getattr(self, 'beta', 'N/A')}\n")
            f.write(f"alpha (weighted_avg): {getattr(self, 'alpha', 'N/A')}\n\n")

            f.write("=== Hardware ===\n")
            f.write(f"GPU: {self.GPU}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"num_workers: {self.num_workers}\n")

        print(f"[INFO] Saved parameter record to: {txt_path}")

    def run(self):
        """
        General function for testing DeepCAD network.
        """
        self.prepare_file()
        self.read_modellist()
        self.read_imglist()
        self.save_yaml_test()
        self.initialize_network()
        self.distribute_GPU()
        self.test()
        self.save_txt_params()

    def prepare_file(self):
        if self.datasets_path[-1] != '/':
            self.datasets_name = self.datasets_path.split("/")[-1]
        else:
            self.datasets_name = self.datasets_path.split("/")[-2]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.output_path = self.output_dir + '//' + 'DataFolderIs_' + self.datasets_name + '_' + current_time + '_ModelFolderIs_' + self.denoise_model
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def set_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))
        self.ngpu = str(self.GPU).count(',') + 1
        self.batch_size = self.ngpu
        print('\033[1;31mTesting parameters -----> \033[0m')
        print(self.__dict__)

    def read_imglist(self):
        im_folder = self.datasets_path
        file_list = list(os.walk(im_folder, topdown=False))[-1][-1]
        split_list = [re.split(r'[._]', filename) for filename in file_list]
        sorted_list = sorted(split_list, key=lambda x: int(x[1]))
        sorted_filenames = ['_'.join(parts[:-1]) + '.' + parts[-1] for parts in sorted_list]
        self.img_list = sorted_filenames
        print('\033[1;31mStacks for processing -----> \033[0m')
        print('Total stack number -----> ', len(self.img_list))
        noise_im_all = []
        for img in self.img_list:
            print(img)
            im_dir = self.datasets_path + '//' + img
            noise_im = tiff.imread(im_dir)
            input_data_type = noise_im.dtype
            noise_im = noise_im.astype(np.float32) * self.scale_factor
            noise_im = noise_im - noise_im.mean()
            noise_im_all.append(noise_im)
        self.noise_im_all = np.array(noise_im_all)
        self.noise_im_mean = np.mean(np.array(noise_im_all), axis=0)

    def read_modellist(self):
        model_path = self.pth_dir + '//' + self.denoise_model
        model_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list.sort()
        count_pth = 0
        for i in range(len(model_list)):
            aaa = model_list[i]
            if '.pth' in aaa:
                count_pth = count_pth + 1
        self.model_list = model_list
        self.model_list_length = count_pth

    def initialize_network(self):
        if self.encode_module == 'SingleConv':
            self.encode_module = SingleConv
        elif self.encode_module == 'DoubleConv':
            self.encode_module = DoubleConv

        denoise_generator_n = Network_3D_Unet(
            in_channels=1, out_channels=1, f_maps=self.fmap,
            encode_module=self.encode_module, final_sigmoid=True,
            conv_kernel=3, pool_type='max'
        )
        denoise_generator_s = Network_3D_Unet(
            in_channels=1, out_channels=1, f_maps=self.fmap,
            encode_module=self.encode_module, final_sigmoid=True,
            conv_kernel=3, pool_type='avg'
        )
        self.local_model_n = denoise_generator_n
        self.local_model_s = denoise_generator_s

    def save_yaml_test(self):
        yaml_name = self.output_path + '//para.yaml'
        para = {'datasets_path': 0, 'test_datasize': 0, 'denoise_model': 0,
                'output_dir': 0, 'pth_dir': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'fmap': 0, 'encode_module': 0,
                'scale_factor': 0, 'overlap_factor': 0}
        para["datasets_path"] = self.datasets_path
        para["denoise_model"] = self.denoise_model
        para["test_datasize"] = self.test_datasize
        para["output_dir"] = self.output_dir
        para["pth_dir"] = self.pth_dir
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["fmap"] = self.fmap
        para["encode_module"] = self.encode_module
        para["scale_factor"] = self.scale_factor
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    
    def save_txt_params(self):
        """Save a human-readable params.txt next to para.yaml.

        用途：
        - 你在資料夾裡一眼看到這次 test 用了哪些參數（尤其 fuse / pipeline）
        - 方便回溯不同結果差異
        """
        try:
            txt_path = os.path.join(self.output_path, "params.txt")
            keys = [
                "datasets_path", "test_datasize", "denoise_model",
                "output_dir", "output_path", "pth_dir",
                "GPU", "ngpu", "batch_size", "num_workers",
                "patch_t", "patch_x", "patch_y", "gap_t", "gap_x", "gap_y", "overlap_factor",
                "fmap", "encode_module", "scale_factor",
                "pipeline_mode", "fuse_mode", "fuse_beta", "fuse_alpha",
                "alpha_min", "alpha_max", "alpha_k",
            ]
            lines = []
            for k in keys:
                if hasattr(self, k):
                    v = getattr(self, k)
                    if callable(v):
                        v = str(v)
                    lines.append(f"{k}: {v}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print("[SAVED TXT]", txt_path)
        except Exception as e:
            print("[WARN] save_txt_params failed:", repr(e))
    def distribute_GPU(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model_n = self.local_model_n.cuda()
            self.local_model_s = self.local_model_s.cuda()
            self.local_model_n = nn.DataParallel(self.local_model_n, device_ids=range(self.ngpu))
            self.local_model_s = nn.DataParallel(self.local_model_s, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def _get_weight_map(self, t: int, h: int, w: int) -> np.ndarray:
        """cache weight map by shape"""
        key = (t, h, w)
        wm = self._weight_cache.get(key, None)
        if wm is None:
            wm = make_weight_map_3d(t, h, w)
            self._weight_cache[key] = wm
        return wm


    # =========================================================
    # <<< NEW >>> Fusion helper
    # <<< WHY >>> 與 train_collection_v2 對齊：fused = S + beta*(N - S)
    #            避免你之前用 (N+S)/2 造成效果與訓練目標不一致、對比變灰
    # =========================================================
    def _fuse(self, output_img_n: np.ndarray, output_img_s: np.ndarray) -> np.ndarray:
        """Fuse N/S outputs into final output (float32).
        - weighted_avg: fused = alpha*N + (1-alpha)*S
        - detail_inject: fused = S + beta*(N - S)
        """
        n = output_img_n.astype(np.float32, copy=False)
        s = output_img_s.astype(np.float32, copy=False)

        mode = getattr(self, 'fuse_mode', 'detail_inject')
        if mode == 'weighted_avg':
            alpha = float(getattr(self, 'fuse_alpha', 0.5))
            return alpha * n + (1.0 - alpha) * s

        beta = float(getattr(self, 'fuse_beta', 0.55))
        return s + beta * (n - s)



    def test(self):
        """
        Pytorch testing workflow

        ---
        <<< MODIFIED >>>
        原本檔案註解寫「不再支援 recombination」，但你現在需要 stitch 回整張 volume，
        並用 weight map 做 overlap blending，所以這裡正式支援 recombination。
        ---
        """

        pth_count = 0
        for pth_index in range(int(self.model_list_length / 2)):
            aaa = self.model_list[pth_index]
            if '.pth' in aaa:
                pth_count = pth_count + 1
                pth_name = self.model_list[pth_index * 2]
                pth_name = pth_name.replace('.pth', '')
                pth_name = pth_name[:-2]
                output_path_name = self.output_path + '//' + pth_name.replace('.pth', '')
                if not os.path.exists(output_path_name):
                    os.mkdir(output_path_name)

                # load model
                for i in range(2):
                    pth_name_i = self.model_list[pth_index * 2 + i]
                    pth_name_i = pth_name_i.replace('.pth', '')
                    if pth_name_i[-1] == "s":
                        model_name_s = self.pth_dir + '//' + self.denoise_model + '//' + self.model_list[pth_index * 2 + i]
                    else:
                        model_name_n = self.pth_dir + '//' + self.denoise_model + '//' + self.model_list[pth_index * 2]

                    print(pth_name_i)

                if isinstance(self.local_model_n, nn.DataParallel):
                    self.local_model_n.module.load_state_dict(torch.load(model_name_n), strict=False)
                    self.local_model_n.eval()
                    self.local_model_s.module.load_state_dict(torch.load(model_name_s), strict=False)
                    self.local_model_s.eval()
                else:
                    self.local_model_n.load_state_dict(torch.load(model_name_n), strict=False)
                    self.local_model_n.eval()
                    self.local_model_s.load_state_dict(torch.load(model_name_s), strict=False)
                    self.local_model_s.eval()

                self.local_model_n.cuda()
                self.local_model_s.cuda()
                self.print_img_name = False

                # test all stacks
                for N in range(len(self.img_list)):
                    name_list, noise_img_n, noise_img_s, coordinate_list, test_im_name, img_mean, input_data_type = \
                        test_preprocess_chooseOne(self, N, self.scale_factor, self.noise_im_all)

                    prev_time = time.time()
                    time_start = time.time()

                    # =========================================================
                    # <<< MODIFIED >>> (NEW) accumulators for weight-map stitching
                    # <<< WHY >>>      原本 denoise_img_n/s 只是最後一塊 patch，沒有 stitch
                    # <<< EFFECT >>>   用 out_acc / w_acc 做 overlap 加權平均
                    # =========================================================
                    out_acc_n = np.zeros(noise_img_n.shape, dtype=np.float32)
                    out_acc_s = np.zeros(noise_img_s.shape, dtype=np.float32)
                    w_acc     = np.zeros(noise_img_n.shape, dtype=np.float32)

                    test_data = testset(name_list, coordinate_list, noise_img_n, noise_img_s)
                    testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers)

                    for iteration, (noise_patch_n, noise_patch_s, single_coordinate) in enumerate(testloader):
                        noise_patch_n = noise_patch_n.cuda()
                        noise_patch_s = noise_patch_s.cuda()

                        real_A_n = Variable(noise_patch_n)
                        real_A_s = Variable(noise_patch_s)

                        with torch.no_grad():
                            fake_B_n = self.local_model_n(real_A_n)
                            fake_B_s = self.local_model_s(real_A_s)

                        # ETA
                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(batches_left * (time.time() - prev_time))
                        time_left = datetime.timedelta(seconds=time_left_seconds)
                        prev_time = time.time()

                        if iteration % 1 == 0:
                            time_end = time.time()
                            time_cost = time_end - time_start
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                                % (
                                    pth_count,
                                    self.model_list_length,
                                    pth_name,
                                    N + 1,
                                    len(self.img_list),
                                    self.img_list[N],
                                    iteration + 1,
                                    len(testloader),
                                    time_cost,
                                    time_left_seconds
                                ), end=' ')

                        if (iteration + 1) % len(testloader) == 0:
                            print('\n', end=' ')

                        # =========================================================
                        # <<< MODIFIED >>>  (FIX) 取出 network output / raw patch
                        # <<< WHY >>>       原本有 output_image 未定義 bug
                        # =========================================================
                        out_n = np.squeeze(fake_B_n.detach().cpu().numpy())
                        out_s = np.squeeze(fake_B_s.detach().cpu().numpy())
                        raw   = np.squeeze(real_A_n.detach().cpu().numpy())

                        # 解析 coordinates（支援 batch）
                        coords = _coords_from_dataloader_batch(single_coordinate)

                        # =========================================================
                        # <<< MODIFIED >>>  (NEW) 支援 batch_size>1 的情況
                        # out_n/out_s/raw squeeze 後可能是：
                        #   (T,H,W) 或 (B,T,H,W)
                        # =========================================================
                        if out_n.ndim == 3:
                            out_n_list = [out_n]
                            out_s_list = [out_s]
                            raw_list   = [raw]
                        elif out_n.ndim == 4:
                            out_n_list = [out_n[b] for b in range(out_n.shape[0])]
                            out_s_list = [out_s[b] for b in range(out_s.shape[0])]
                            raw_list   = [raw[b]   for b in range(raw.shape[0])]
                        else:
                            raise ValueError(f"Unexpected out_n.ndim={out_n.ndim}, shape={out_n.shape}")

                        # =========================================================
                        # <<< MODIFIED >>>  (NEW) stitch each patch with weight map
                        # <<< WHY >>>       你要用 weight map，重疊區域做加權平均
                        # =========================================================
                        for b in range(len(out_n_list)):
                            coord = coords[b] if b < len(coords) else coords[-1]
                            init_h, end_h = coord['init_h'], coord['end_h']
                            init_w, end_w = coord['init_w'], coord['end_w']
                            init_s, end_s = coord['init_s'], coord['end_s']

                            patch_out_n = out_n_list[b].astype(np.float32)
                            patch_out_s = out_s_list[b].astype(np.float32)
                            patch_raw   = raw_list[b].astype(np.float32)

                                                        # =========================================================
                            # Output interpretation (train/test alignment)
                            #
                            # pipeline_mode:
                            #   - 'train_v2' (default): 直接使用模型輸出當作 denoised（與 train_collection_v2 完全一致）
                            #         patch_dn = out_n
                            #         patch_ds = out_s
                            #
                            #   - 'legacy_energy': 舊版「加回 mean + 能量比例校正」（保底/對照用，已加 NaN-safe）
                            #
                            #   - 'consistent_corr': 舊版「x + alpha*tanh(out) 並 clip 到[0,1]」
                            #         ⚠️ 注意：這個模式和 train_collection_v2 的訓練語意不同，只是方便你做對照
                            # =========================================================
                            mode_pm = getattr(self, 'pipeline_mode', 'train_v2')

                            if mode_pm == 'legacy_energy':
                                # --- LEGACY: add mean + energy scaling (NaN-safe guards) ---
                                patch_out_n = patch_out_n + img_mean
                                patch_out_s = patch_out_s + img_mean
                                patch_raw   = patch_raw + img_mean

                                eps = 1e-8
                                ratio_n = np.sum(patch_raw) / (np.sum(patch_out_n) + eps)
                                ratio_s = np.sum(patch_raw) / (np.sum(patch_out_s) + eps)
                                ratio_n = max(float(ratio_n), 0.0)
                                ratio_s = max(float(ratio_s), 0.0)
                                scale_n = np.sqrt(ratio_n)
                                scale_s = np.sqrt(ratio_s)

                                patch_dn = patch_out_n * scale_n
                                patch_ds = patch_out_s * scale_s

                            elif mode_pm == 'consistent_corr':
                                # --- CONSISTENT CORRECTION (對照用) ---
                                x_in = patch_raw.astype(np.float32)

                                # N branch
                                out_n_f = patch_out_n.astype(np.float32)
                                scale = np.median(np.abs(out_n_f)).astype(np.float32)
                                alpha = float(self.alpha_k) / float(scale + 1e-6)
                                alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
                                corr  = alpha * np.tanh(out_n_f)
                                patch_dn = np.clip(x_in + corr, 0.0, 1.0).astype(np.float32)

                                # S branch
                                out_s_f = patch_out_s.astype(np.float32)
                                scale = np.median(np.abs(out_s_f)).astype(np.float32)
                                alpha = float(self.alpha_k) / float(scale + 1e-6)
                                alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
                                corr  = alpha * np.tanh(out_s_f)
                                patch_ds = np.clip(x_in + corr, 0.0, 1.0).astype(np.float32)

                            else:
                                # --- TRAIN_V2 (default): direct denoised output ---
                                patch_dn = patch_out_n.astype(np.float32)
                                patch_ds = patch_out_s.astype(np.float32)

                            # --- weight map (依 patch 真實尺寸生成) ---
                            pt, ph, pw = patch_dn.shape
                            wm = self._get_weight_map(pt, ph, pw)

                            # --- accumulate ---
                            out_acc_n[init_s:end_s, init_h:end_h, init_w:end_w] += patch_dn * wm
                            out_acc_s[init_s:end_s, init_h:end_h, init_w:end_w] += patch_ds * wm
                            w_acc[init_s:end_s, init_h:end_h, init_w:end_w]     += wm

                    # =========================================================
                    # <<< MODIFIED >>> (NEW) finalize stitched result
                    # <<< WHY >>>      用累積值/權重和做平均，得到整張 volume
                    # =========================================================
                    eps = 1e-8
                    output_img_n = (out_acc_n / (w_acc + eps)).astype(np.float32)
                    output_img_s = (out_acc_s / (w_acc + eps)).astype(np.float32)

                    # =========================================================
                    # Save inference image  (FIXED: always save when enabled)
                    # =========================================================
                    if self.save_test_images_per_epoch:
                        # 1) fuse (預設 detail_inject 與 train_collection_v2 一致)
                        output_img_fused = self._fuse(output_img_n, output_img_s).astype(np.float32)

                        # 2) decide saving domain
                        mode_pm = getattr(self, 'pipeline_mode', 'train_v2')
                        if mode_pm == 'train_v2':
                            # train_v2：模型輸出在 mean-centered domain -> 存檔前把 mean 加回去
                            output_save = output_img_fused + float(img_mean)
                        else:
                            # legacy_energy：前面已加回 mean/做過縮放；consistent_corr：本來就在 [0,1]
                            output_save = output_img_fused

                        # 3) cast back to original dtype for writing
                        output_save = np.nan_to_num(output_save, nan=0.0, posinf=0.0, neginf=0.0)

                        if mode_pm == 'consistent_corr':
                            # 以 [0,1] 視覺化輸出
                            output_01 = np.clip(output_save, 0.0, 1.0)
                            output_img = np.round(output_01 * 65535.0).astype(np.uint16)
                        else:
                            # 依原始 dtype 做 clip/cast
                            if input_data_type == np.uint16 or str(input_data_type) == 'uint16':
                                output_img = np.clip(output_save, 0, 65535).astype(np.uint16)
                            elif input_data_type == np.int16 or str(input_data_type) == 'int16':
                                output_img = np.clip(output_save, -32768, 32767).astype(np.int16)
                            elif input_data_type == np.uint8 or str(input_data_type) == 'uint8':
                                output_img = np.clip(output_save, 0, 255).astype(np.uint8)
                            else:
                                # fallback：存 uint16
                                output_img = np.clip(output_save, 0, 65535).astype(np.uint16)

                        # 4) write TIFF
                        result_name = (
                            output_path_name
                            + '//' + self.img_list[N].replace('.tiff', '').replace('.tif', '')
                            + '_denoised.tiff'
                        )
                        tiff.imwrite(result_name, output_img)
                        print("[SAVED TIFF]", result_name, output_img.shape, output_img.dtype)
        print('Test finished. Save all results to disk.')
