import os
import cv2
import glob
import math
import numpy as np
from PIL import Image, ImageEnhance
from enum import Enum
from typing import Dict
from torch.utils.data import Dataset
from torchvision.datasets import ETH3DStereo, InStereo2k
from read_pfm import read_pfm

class Augmentor:
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, left_img, right_img, left_disp):
        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        disp_mask = ((left_disp < float(self.max_disp)) & (left_disp > 0)).astype("float32")

        return left_img, right_img, left_disp, disp_mask


class CREStereoDataset(Dataset):
    def __init__(self, root, image_height=384, image_width=512, max_disp=256, train_mode=True):
        super().__init__()
        self.imgs = glob.glob(os.path.join(root, "**/*_left.jpg"), recursive=True)
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.augmentor = Augmentor(
            image_height=image_height,
            image_width=image_width,
            max_disp=max_disp,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
        self.train_mode = train_mode

    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32

    def __getitem__(self, index):
        # find path
        left_path = self.imgs[index]
        prefix = left_path[: left_path.rfind("_")]
        right_path = prefix + "_right.jpg"
        left_disp_path = prefix + "_left.disp.png"
        right_disp_path = prefix + "_right.disp.png"

        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)

        if self.rng.binomial(1, 0.5):
            left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
            left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
        left_disp[left_disp == np.inf] = 0

        if self.train_mode:
            # augmentaion
            left_img, right_img, left_disp, disp_mask = self.augmentor(
                left_img, right_img, left_disp
            )
        else:
            # validation mode
            left_img = cv2.resize(left_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            resize_x_scale = float(self.image_width) / left_disp.shape[1]
            left_disp = cv2.resize(left_disp, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR) * resize_x_scale
            disp_mask = (left_disp < float(self.max_disp)) & (left_disp > 0).astype("float32")

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.imgs)

class Eth3dDataset(Dataset):
    """ETH3D Stereo Dataset.
    The dataset is expected to be organized as follows:
    <root>
        |-- two_view_training
            |-- scene_1
                |-- im0.png
                |-- im1.png
            |-- scene_2
                |-- im0.png
                |-- im1.png
        |-- two_view_training_gt
            |-- scene_1
                |-- disp0GT.pfm
                |-- mask0nocc.png
            |-- scene_2
                |-- disp0GT.pfm
                |-- mask0nocc.png
    """
    def __init__(self, root, image_height=384, image_width=512, max_disp=256, train_mode=True):
        super().__init__()
        self.imgs = glob.glob(os.path.join(root, "**/im0.png"), recursive=True)
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.augmentor = Augmentor(
            image_height=image_height,
            image_width=image_width,
            max_disp=max_disp,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
        self.train_mode = train_mode

    def get_disp(self, path):
        return read_pfm(path)

    def __getitem__(self, index):
        left_path = self.imgs[index]
        prefix = left_path[: left_path.rfind("im0.png")]
        right_path = prefix + "im1.png"
        prefix = prefix.replace("two_view_training", "two_view_training_gt")
        disp_path = prefix + "disp0GT.pfm"
        disp_mask_path = prefix + "mask0nocc.png"

        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        disp_mask = cv2.imread(disp_mask_path, cv2.IMREAD_GRAYSCALE)
        disp_img = self.get_disp(disp_path)
        disp_img[disp_img == np.inf] = 0

        if self.train_mode:
            # augmentaion
            left_img, right_img, disp_img, disp_mask = self.augmentor(
                left_img, right_img, disp_img
            )
        else:
            # validation mode
            left_img = cv2.resize(left_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            resize_x_scale = float(self.image_width) / float(disp_mask.shape[1])
            disp_img = cv2.resize(disp_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR) * resize_x_scale
            disp_mask = cv2.resize(disp_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            disp_mask = (disp_mask > 0).astype("float32")

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": disp_img,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.imgs)


class DataSetType(Enum):
    CRESTREREO = CREStereoDataset
    ETH3D = Eth3dDataset

def DataSetWrapper(
        dataset_name: str,
        data_dir: str,
        image_height: int = 384,
        image_width: int = 512,
        max_disp: int = 256,
        train_mode: bool = True):
    try:
        dataset_type = DataSetType[dataset_name.upper()]
        dataset_class = dataset_type.value
        return dataset_class(root=data_dir, image_height=image_height, image_width=image_width,
                             max_disp=max_disp, train_mode=train_mode)
    except:
      print(f'ERROR: {dataset_name} is not a valid dataset type.')
      return None

class MixedDataset(Dataset):
    """Mixed dataset from multiple datasets. Use torchvision datasets as the base datasets.
    """
    def __init__(self,
                dataset_roots: Dict[str, str] = {},
                image_height: int = 384,
                image_width: int = 512,
                max_disp: int = 256,
                train_mode: bool = True):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.augmentor = Augmentor(
            image_height=image_height,
            image_width=image_width,
            max_disp=max_disp,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
        self.train_mode = train_mode

        self.supported_dataset_weights = {
            "ETH3D": 1,
            "InStereo2k": 0,
        }

        self.dataset_weights = {}
        self.dataset_lens = {}
        self.datasets = {}
        for dataset_name, dataset_root in dataset_roots.items():
            if dataset_name not in self.supported_dataset_weights:
                raise ValueError(f"Invalid dataset name: {dataset_name}")
            if self.supported_dataset_weights[dataset_name] < 1e-5:
                print(f"Skipping {dataset_name} dataset with weight {self.supported_dataset_weights[dataset_name]}")
                continue

            self.dataset_weights[dataset_name] = self.supported_dataset_weights[dataset_name]
            if dataset_name == "ETH3D":
                # spilt = "train" if self.train_mode else "test"
                spilt = "train"  # no gt for test set, use train instead
                self.datasets[dataset_name] = ETH3DStereo(root=dataset_root, split=spilt)
            elif dataset_name == "InStereo2k":
                spilt = "train" if self.train_mode else "test"
                self.datasets[dataset_name] = InStereo2k(root=dataset_root, split=spilt)
            self.dataset_lens[dataset_name] = math.floor(len(self.datasets[dataset_name]) * self.dataset_weights[dataset_name])
            print(f"Loaded {dataset_name} dataset with {self.dataset_lens[dataset_name]} samples with weight {self.dataset_weights[dataset_name]}")

        self.index_mapping = []
        for dataset_name in self.datasets.keys():
            total_samples = len(self.datasets[dataset_name])
            needed_samples = self.dataset_lens[dataset_name]
            if needed_samples > 0:
                idxs = self.rng.choice(total_samples, size=needed_samples, replace=False)
                self.index_mapping.extend([(dataset_name, idx) for idx in idxs])


    def __getitem__(self, index):
        if index >= len(self.index_mapping) or index < 0:
            raise IndexError(f"Index {index} is out of range for MixedDataset with {len(self.index_mapping)} samples")

        dataset_name, sub_index = self.index_mapping[index]
        item = self.datasets[dataset_name][sub_index]
        left_img = np.array(item[0])
        right_img = np.array(item[1])
        left_disp = np.array(item[2])[0] if item[2][0] is not None else None
        left_disp[left_disp == np.inf] = 0
        if len(item) == 4:
            disp_mask = np.array(item[3])
        else:
            disp_mask = None

        if self.train_mode:
            # augmentaion
            left_img, right_img, left_disp, disp_mask = self.augmentor(
                left_img, right_img, left_disp
            )
        else:
            # validation mode
            left_img = cv2.resize(left_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            if left_disp is not None:
                original_width = left_disp.shape[1]
                resize_x_scale = float(self.image_width) / original_width
                left_disp = cv2.resize(left_disp, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR) * resize_x_scale
            if disp_mask is not None:
                disp_mask = disp_mask.astype(np.uint8)
                disp_mask = cv2.resize(disp_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                disp_mask = (disp_mask > 0).astype("float32")
            elif left_disp is not None:
                disp_mask = ((left_disp < float(self.max_disp)) & (left_disp > 0)).astype("float32")

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return sum(self.dataset_lens.values())
