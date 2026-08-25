# @Time    : 2023/8/29 02:01
# @Author  : zhangchenming
import random
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from PIL import Image
from torchvision.transforms import ColorJitter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TransposeImage(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                if k == 'super_pixel_label':
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.int32)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.bool)
                else:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample


class NormalizeImage(object):
    def __init__(self, config):
        self.mean = config.MEAN
        self.std = config.STD

    def __call__(self, sample):
        sample['left_ori'] = sample['left'].clone()
        sample['left'] = normalize(sample['left'] / 255.0, self.mean, self.std)
        sample['right_ori'] = sample['right'].clone()
        sample['right'] = normalize(sample['right'] / 255.0, self.mean, self.std)
        return sample


class RandomCrop(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.base_size = config.SIZE
        self.y_jitter = config.get('Y_JITTER', False)

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        # crop_height = min(height, crop_height)
        # crop_width = min(width, crop_width)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = random.randint(n_pixels, height - crop_height - n_pixels)
        x1 = random.randint(0, width - crop_width)
        y2 = y1 + np.random.randint(-n_pixels, n_pixels + 1)

        for k in sample.keys():
            if k in ['right']:
                sample[k] = sample[k][y2: y2 + crop_height, x1: x1 + crop_width]
            elif k in ['pos']: #iinet
                sample[k] = sample[k][:, y1: y1 + crop_height, x1: x1 + crop_width]
            else:
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        if 'disp' in sample:
            disp = sample['disp']
            valid = np.isfinite(disp) & (disp > 0)
            if valid.mean() < 0.01:   # 유효 픽셀 1% 미만이면 재시도/스킵
                sample['_bad_crop'] = True

        return sample


class RandomScale(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB
        self.stretch_prob = config.STRETCH_PROB

        self.max_stretch = 0.2

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    sample[k] = sample[k] * scale_x

        return sample


class RandomSparseScale(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right']:
                    sample[k] = self.sparse_disp_map_reisze(sample[k], fx=scale_x, fy=scale_y)

        return sample

    @staticmethod
    def sparse_disp_map_reisze(disp, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        valid = disp > 0.0
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)  # 坐标[w, h]

        disp = disp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1)
        coords = coords[valid]  # disp > 0 的坐标
        disp = disp[valid]  # disp > 0 的值
        coords = coords * [fx, fy]  # resize 后的坐标
        disp = disp * fx  # risize 后的值

        xx = np.round(coords[:, 0]).astype(np.int32)
        yy = np.round(coords[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp = disp[v]

        resized_disp = np.zeros([ht1, wd1], dtype=np.float32)
        resized_disp[yy, xx] = disp

        return resized_disp


class RandomErase(object):
    def __init__(self, config):
        self.eraser_aug_prob = config.PROB
        self.max_erase_time = config.MAX_TIME
        self.bounds = config.BOUNDS

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        ht, wd = img1.shape[:2]
        if 'super_pixel_label' in sample:
            occ_mask_2 = np.zeros((ht, wd), dtype=bool)
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, self.max_erase_time + 1)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                if 'super_pixel_label' in sample:
                    occ_mask_2 = np.zeros((ht, wd), dtype=bool)

        sample['left'] = img1
        sample['right'] = img2
        if 'super_pixel_label' in sample:
            sample['occ_mask_2'] = occ_mask_2
        return sample


class StereoColorJitter(object):
    def __init__(self, config):
        self.brightness = list(config.BRIGHTNESS)
        self.contrast = list(config.CONTRAST)
        self.saturation = list(config.SATURATION)
        if isinstance(config.HUE, float):
            config.HUE = [-config.HUE, config.HUE]
        self.hue = list(config.HUE)
        self.asymmetric_color_aug_prob = config.ASYMMETRIC_PROB
        self.color_jitter = ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                        saturation=self.saturation, hue=[x / 3.14 for x in self.hue])

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.color_jitter(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.color_jitter(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class RightTopPad(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        return sample


class DivisiblePad(object):
    def __init__(self, config):
        self.by = config.BY
        self.mode = config.get('MODE', 'tr')

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = self.by - h % self.by
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = self.by - w % self.by
        else:
            pad_w = 0

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no DivisiblePad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
        return sample


class RandomFlip(object):
    def __init__(self, config):
        self.config = config
        self.flip_type = config.FLIP_TYPE
        self.prob = config.PROB

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        disp_right = sample['disp_right']

        if np.random.rand() < self.prob and self.flip_type == 'horizontal':  # 水平翻转
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp[:, ::-1] * -1.0)

        if np.random.rand() < self.prob and self.flip_type == 'horizontal_swap':  # 水平翻转并交换
            tmp = np.ascontiguousarray(img1[:, ::-1])
            img1 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp_right[:, ::-1])
            img2 = tmp

        if np.random.rand() < self.prob and self.flip_type == 'vertical':  # 垂直翻转
            img1 = np.ascontiguousarray(img1[::-1, :])
            img2 = np.ascontiguousarray(img2[::-1, :])
            disp = np.ascontiguousarray(disp[::-1, :])

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


class RightBottomCrop(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        for k in sample.keys():
            sample[k] = sample[k][h - crop_h:, w - crop_w:]
        return sample


class CropOrPad(object):
    def __init__(self, config):
        self.size = config.SIZE
        self.crop_fn = RightBottomCrop(config)
        self.pad_fn = RightTopPad(config)

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            sample = self.pad_fn(sample)
        else:
            sample = self.crop_fn(sample)

        return sample


def _ensure_hwc(img):
    # img: np.ndarray or torch.Tensor
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3:  # CHW → HWC
            img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:  # gray → HWC(…,1)
        img = img[..., None]
    return img  # HWC

class AddGradientMap:
    def __init__(self, config):
        self.ks = int(config.KSIZE)
        self.key_out = config.KEY_OUT
    def __call__(self, sample):
        left = _ensure_hwc(sample["left"])
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.shape[2] >= 3 else left[...,0]
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.ks)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.ks)
        mag = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        sample[self.key_out] = mag  # HxW (단일 채널)
        return sample

class AddLaplacianMap:
    def __init__(self, config):
        self.ks = int(config.KSIZE); self.key_out = config.KEY_OUT
    def __call__(self, sample):
        left = _ensure_hwc(sample["left"])
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.shape[2] >= 3 else left[...,0]
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=self.ks)
        sample[self.key_out] = lap.astype(np.float32)   # HxW
        return sample

class AddRepExpansion:
    """
    보조 채널을 'rep_feats'(HxWxK) 키로 추가. left/right는 절대 수정 X.
    게이트가 비활성/결과가 비어도 최소 1채널(zeros)을 보장해서 (H,W,0) 방지.
    """
    def __init__(self, config):
        self.ks = int(config.KERNEL)
        self.edge_gate = bool(config.EDGE_GATE)
        self.texless_gate = bool(config.TEXLESS_GATE)
        self.key_out = config.KEY_OUT

    def __call__(self, sample):
        left = _ensure_hwc(sample["left"])
        H, W, _ = left.shape
        feats = []

        # 예: 간단한 엣지/텍스쳐 약식 특징 (실사용은 기존 로직 연결)
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.shape[2] >= 3 else left[...,0]
        if self.edge_gate:
            e = cv2.Canny((gray*255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
            feats.append(e[...,None])
        if self.texless_gate:
            blur = cv2.GaussianBlur(gray, (self.ks|1, self.ks|1), 0)
            tless = (np.abs(gray - blur) < 0.02).astype(np.float32)
            feats.append(tless[...,None])

        if len(feats) == 0:
            rep = np.zeros((H, W, 1), dtype=np.float32)  # (H,W,0) 방지
        else:
            rep = np.concatenate(feats, axis=2)  # HxWxK
        sample[self.key_out] = rep
        return sample
    

class AlignSize:
    """
    Ensure all tensors (left/right/disp) have identical spatial size.
    Pads or crops minimally to match the largest among them.
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        keys = ["left", "right", "disp"]
        Hs, Ws = [], []

        # --- 안전 검사 ---
        if not isinstance(sample, dict):
            return sample

        for k in keys:
            if k in sample and isinstance(sample[k], np.ndarray):
                if sample[k].size == 0:  # 빈 배열
                    continue
                Hs.append(sample[k].shape[0])
                Ws.append(sample[k].shape[1])

        # --- 만약 모든 값이 비었으면 그대로 반환 ---
        if len(Hs) == 0 or len(Ws) == 0:
            return sample

        Ht, Wt = max(Hs), max(Ws)

        for k in keys:
            if k in sample and isinstance(sample[k], np.ndarray):
                img = sample[k]
                h, w = img.shape[:2]
                pad_h = Ht - h
                pad_w = Wt - w
                if pad_h > 0 or pad_w > 0:
                    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                elif pad_h < 0 or pad_w < 0:
                    img = img[:Ht, :Wt]
                sample[k] = img

        return sample
    

class AlignSizeSafe:
    """
    AlignSizeSafe
    --------------
    - 입력 이미지/맵들의 해상도를 동일하게 맞춘다.
    - align_to='max'이면 현재 샘플의 left/right/disp 중 가장 큰 크기에 맞춰 padding.
    - align_to=(H,W) tuple이면 지정된 크기에 맞춤.
    - padding은 오른쪽/아래쪽 기준으로 zero-padding.
    """
    def __init__(self, config):
        self.align_to = config.ALIGN_TO

    def _pad_to(self, img, Ht, Wt):
        """오른쪽/아래쪽 기준으로 zero padding"""
        if img is None:
            return None
        H, W = img.shape[:2]
        pad_h = max(Ht - H, 0)
        pad_w = max(Wt - W, 0)
        if pad_h == 0 and pad_w == 0:
            return img
        if img.ndim == 2:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
        else:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    def __call__(self, sample):
        keys = ["left", "right", "disp"]
        shapes = [sample[k].shape[:2] for k in keys if k in sample and sample[k] is not None]

        # align target 결정
        if self.align_to == "max":
            Ht = max([s[0] for s in shapes])
            Wt = max([s[1] for s in shapes])
        elif isinstance(self.align_to, (list, tuple)):
            Ht, Wt = self.align_to
        else:
            raise ValueError(f"Invalid align_to: {self.align_to}")

        # 실제 padding 수행
        for k in keys:
            if k in sample and sample[k] is not None:
                sample[k] = self._pad_to(sample[k], Ht, Wt)

        return sample
    

class ShapeLogger:
    def __init__(self, config):
        self.prefix = config.PREFIX

    def __call__(self, sample):
        import numpy as np
        for k in ["left", "right"]:
            if k not in sample:
                continue
            arr = sample[k]
            if isinstance(arr, np.ndarray):
                print(f"[{self.prefix}] {k}: np.ndarray {arr.shape}")
            elif hasattr(arr, 'shape'):
                print(f"[{self.prefix}] {k}: tensor {tuple(arr.shape)}")
            else:
                print(f"[{self.prefix}] {k}: {type(arr)}")
        return sample
    

class TensorShapeCheck:
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        import torch
        for k in ["left", "right"]:
            if k in sample and isinstance(sample[k], torch.Tensor):
                t = sample[k]
                if t.numel() == 0 or t.shape[0] == 0:
                    print(f"[ERR] {k} tensor is empty! shape={tuple(t.shape)}")
                else:
                    print(f"[OK] {k} tensor shape={tuple(t.shape)}")
        return sample
    

class ResizeToSafeShape:
    """
    KITTI12 / KITTI15 크기 불일치 해결용.
    두 데이터셋의 원본 해상도를 동일한 안전 크기(W_target)로 리사이즈.
    aspect ratio를 유지하며 W_target 기준으로 스케일링.
    """
    def __init__(self, config):
        self.Ht = config.H_TARGET
        self.Wt = config.W_TARGET

    def __call__(self, sample):
        for key in ["left", "right", "disp"]:
            if key in sample and sample[key] is not None:
                img = sample[key]
                # numpy 변환 강제
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))
                if img.ndim == 2:
                    sample[key] = cv2.resize(img, (self.Wt, self.Ht), interpolation=cv2.INTER_NEAREST)
                else:
                    sample[key] = cv2.resize(img, (self.Wt, self.Ht), interpolation=cv2.INTER_LINEAR)
        return sample
    
# IGEV++ 전용
class InputPadder(object):
    """
    Pads images so that H, W are divisible by `DIVIS_BY` (default: 8).
    - mode='sintel'  : 상하/좌우 대칭 패딩
    - mode='kitti' 등: 위 0, 아래로만 패딩 (좌우는 대칭)

    config 예시:
        TRANSFORM:
            - NAME: InputPadder
              DIVIS_BY: 8
              MODE: sintel
    """
    def __init__(self, config):
        self.divis_by = config.get('DIVIS_BY', 8)
        self.mode = config.get('MODE', 'sintel')

    def __call__(self, sample):
        # 기준은 left 이미지의 크기
        h, w = sample['left'].shape[:2]

        # pad_ht, pad_wd 계산 (원작자 InputPadder와 동일)
        pad_ht = (((h // self.divis_by) + 1) * self.divis_by - h) % self.divis_by
        pad_wd = (((w // self.divis_by) + 1) * self.divis_by - w) % self.divis_by

        # 이미 딱 떨어지면 그대로 반환
        if pad_ht == 0 and pad_wd == 0:
            sample['_pad'] = (0, 0, 0, 0)  # top, bottom, left, right
            return sample

        if self.mode == 'sintel':
            # 상하/좌우 대칭 패딩
            pad_left  = pad_wd // 2
            pad_right = pad_wd - pad_left
            pad_top   = pad_ht // 2
            pad_bot   = pad_ht - pad_top
        else:
            # 위는 0, 아래로만 패딩 (KITTI 스타일)
            pad_left  = pad_wd // 2
            pad_right = pad_wd - pad_left
            pad_top   = 0
            pad_bot   = pad_ht

        # 나중에 unpad 할 수 있도록 저장
        sample['_pad'] = (pad_top, pad_bot, pad_left, pad_right)

        # numpy의 replicate padding == mode='edge'
        for k, v in list(sample.items()):
            if k.startswith('_'):   # 메타데이터는 스킵
                continue
            arr = v
            if not isinstance(arr, np.ndarray):
                continue

            if arr.ndim == 2:
                # [H, W]
                sample[k] = np.pad(
                    arr,
                    ((pad_top, pad_bot), (pad_left, pad_right)),
                    mode='edge'
                )
            elif arr.ndim == 3:
                # [H, W, C]
                sample[k] = np.pad(
                    arr,
                    ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
                    mode='edge'
                )
            elif arr.ndim == 4:
                # 혹시 [C, H, W, ?] 같은 특수 케이스가 있으면 여기에 맞춰서 수정
                # 여기서는 [C, H, W] (예: pos)만 따로 처리
                # pos는 RandomCrop에서 [C, H, W] 기준이라 그 케이스만 지원
                # 필요 없으면 이 블록은 제거해도 됨.
                pass

        return sample

def unpad_prediction(pred, pad):
    """
    pred: [B, C, H, W] 또는 [H, W] / [H, W, C] 등
    pad: (pad_top, pad_bot, pad_left, pad_right)
    """
    pad_top, pad_bot, pad_left, pad_right = pad

    if pred.ndim == 4:
        # [B, C, H, W]
        return pred[..., pad_top:pred.shape[-2]-pad_bot,
                    pad_left:pred.shape[-1]-pad_right]
    elif pred.ndim == 3:
        # [H, W, C]
        return pred[pad_top:pred.shape[0]-pad_bot,
                    pad_left:pred.shape[1]-pad_right]
    elif pred.ndim == 2:
        # [H, W]
        return pred[pad_top:pred.shape[0]-pad_bot,
                    pad_left:pred.shape[1]-pad_right]
    else:
        return pred