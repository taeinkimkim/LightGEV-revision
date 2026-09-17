import os
import numpy as np
from pathlib import Path
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres
import time


class SceneFlowDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

        # 기존 옵션 유지
        self.retrun_pos = getattr(self.data_info, 'RETURN_POS', False)
        self.retrun_super_pixel = getattr(self.data_info, 'RETURN_SUPER_PIXEL', False)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP

        self.cleanpass = self.data_info.CLEANPASS  # cleanpass or finalpass

    def __getitem__(self, idx):
        
        # data_list[idx] = "train/000000.npz" 형태
        npz_rel_path = self.data_list[idx][0]    # txt에서 한 줄당 1개 항목만 사용
        npz_path = os.path.join(self.root, npz_rel_path)

        # ----- NPZ 로딩 -----
        data = np.load(npz_path)

        if self.cleanpass:
            left_img = data['left_clean']     # [H, W, 3] float32
            right_img = data['right_clean']   # [H, W, 3] float32
        else:
            left_img = data['left_final']     # [H, W, 3] float32
            right_img = data['right_final']   # [H, W, 3] float32
        disp_img = data['disp']     # [H, W] float32

        # (기존 PNG + PFM 로딩 단계 삭제됨)
        # from PIL import Image
        # from pathlib import Path
        # from stereo.datasets.dataset_utils.readpfm import readpfm

        # item = self.data_list[idx]
        # full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        # left_img_path, right_img_path, disp_img_path = full_paths
        # left_img = Image.open(left_img_path).convert('RGB')
        # left_img = np.array(left_img, dtype=np.float32)
        # right_img = Image.open(right_img_path).convert('RGB')
        # right_img = np.array(right_img, dtype=np.float32)
        # disp_img = readpfm(disp_img_path)[0].astype(np.float32)

        # ----- SuperPixel 처리 -----
        if self.retrun_super_pixel:
            # 기존 코드 유지 (left_img 이용)
            super_pixel_label = Path(self.root).parent.joinpath(
                'SuperPixelLabel/SceneFlow', npz_rel_path
            )
            super_pixel_label = str(super_pixel_label).replace('.npz', '_lsc_lbl.png')

            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)

            # Superpixel 파일 없으면 생성
            if not os.path.exists(super_pixel_label):
                import cv2
                from cv2 import ximgproc
                img_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = ximgproc.createSuperpixelLSC(img_bgr, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                cv2.imwrite(super_pixel_label, label.astype(np.uint16))

            import cv2
            super_pixel_label = cv2.imread(
                super_pixel_label, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ).astype(np.int32)

        # ----- Right disparity(optional) -----
        if self.return_right_disp:
            # 기존: PFM 파일 읽기
            # 수정: 우리는 npz에 disp_right 넣지 않았으므로,
            # SceneFlow는 보통 right disp 필요 없으므로 pass.
            # 필요하다면 npz 전처리 단계에서 같이 저장해야 함.
            raise NotImplementedError(
                "RETURN_RIGHT_DISP=True이면 npz에 disp_right 만들어 저장해야 합니다."
            )

        # ----- sample 패킹 -----
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }

        # superpixel + occ_mask
        if self.retrun_super_pixel and self.mode == 'training':
            occ_mask = np.zeros_like(disp_img, dtype=bool)
            sample['occ_mask'] = occ_mask
            sample['super_pixel_label'] = super_pixel_label

        # 좌표 pos
        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(
                800, left_img.shape[1], left_img.shape[0]
            )

        # transform
        sample = self.transform(sample)
        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = npz_path

        return sample