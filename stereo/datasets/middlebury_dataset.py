import os
import numpy as np
import cv2
from PIL import Image
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


class MiddleburyDataset(DatasetTemplate):
    """Middlebury 2014.

    원본이 ~2000x2900 이라 그대로는 평가가 무겁다. DATA_INFOS 의 SCALE 로 축소
    배율을 지정한다 (기본 0.25 = 기존 동작). disparity 는 x 방향 길이이므로
    이미지와 같은 배율로 값까지 곱해 줘야 한다.

    주의: SCALE 을 키우면 GT disparity 도 그만큼 커진다. 원본 1배의 최대 GT 는
    1074 px 이고 MODEL.MAX_DISP=192 를 크게 넘으므로, 범위 밖 GT 를 어떻게
    다룰지(`--gt_out_of_range`) 를 함께 정해야 수치가 해석 가능하다.
    """

    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

        self.scale = float(getattr(self.data_info, 'SCALE', 0.25))

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
        # Resize (Middlebury 원본은 ~2000x2900 으로 너무 크다)
        # -------------------------------
        scale = self.scale
        H, W = disp_img.shape

        if scale == 1.0:
            left_img_resized = left_img
            right_img_resized = right_img
            disp_resized = disp_img
        else:
            target = (int(W * scale), int(H * scale))   # cv2 는 (w, h) 순서

            left_img_resized = cv2.resize(left_img, target, interpolation=cv2.INTER_AREA)
            right_img_resized = cv2.resize(right_img, target, interpolation=cv2.INTER_AREA)

            # disparity 는 보간하면 경계에서 없는 값이 생기므로 NEAREST,
            # 그리고 x 방향 길이이므로 값 자체도 같은 배율로 줄인다.
            disp_resized = cv2.resize(disp_img, target, interpolation=cv2.INTER_NEAREST)
            disp_resized = disp_resized * scale

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