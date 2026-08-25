import os
import numpy as np
import cv2
from PIL import Image
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


class MiddleburyDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths

        # --- Load images ---
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        # --- Load disparity ---
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        disp_img[disp_img == np.inf] = 0

        # -------------------------------
        # Resize to 1/4 (Middlebury too large)
        # -------------------------------
        scale = 0.25  # 1/4 scaling
        H, W = disp_img.shape

        # Resize images (RGB)
        left_img_resized = cv2.resize(left_img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        right_img_resized = cv2.resize(right_img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

        # Resize disparity and scale disparity values
        disp_resized = cv2.resize(disp_img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)
        disp_resized = disp_resized * scale  # disparity must be scaled too!

        sample = {
            'left': left_img_resized,   # [h, w, 3]
            'right': right_img_resized, # [h, w, 3]
            'disp': disp_resized,       # [h, w]
        }

        # Apply transform (crop, normalize, to tensor...)
        sample = self.transform(sample)

        sample['index'] = idx
        sample['name'] = left_img_path

        return sample