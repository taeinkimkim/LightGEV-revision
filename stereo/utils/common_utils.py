# @Time    : 2023/8/28 22:28
# @Author  : zhangchenming
import os
import yaml
import random
import shutil
import numpy as np
import torch
from torch.nn import Parameter
import logging
import inspect
from easydict import EasyDict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_valid_args(obj, input_args, free_keys=None):
    if free_keys is None:
        free_keys = []
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        k = k.lower()
        if k[0] == 't' and k[1] == '_':
            k = 'T' + k[1:] 
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys:
        print("Find Unexpected Args(%s) in the Configuration of - %s -" % (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def backup_source_code(backup_dir):
    # 子文件夹下的同名也会被忽略
    ignore_hidden = shutil.ignore_patterns(
        ".idea", ".git*", "*pycache*",
        "cfgs", "data", "output")

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree('.', backup_dir, ignore=ignore_hidden)
    # os.system("chmod -R g+w {}".format(backup_dir))


def log_configs(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            logger.info('----------- %s -----------' % key)
            log_configs(cfgs[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def save_checkpoint(model, optimizer, scheduler, scaler, is_dist, epoch, filename='checkpoint'):
    if is_dist:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    optim_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()

    state = {'epoch': epoch,
             'model_state': model_state,
             'optimizer_state': optim_state,
             'scheduler_state': scheduler_state,
             'scaler_state': scaler_state}
    torch.save(state, filename)


def freeze_bn(module):
    """Freeze the batch normalization layers."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    return module


# ------------ helpers ------------
_COMMON_KEYS = [
    "model_state", "model_state_dict", "state_dict",
    "model", "net", "network",
    "ema", "ema_state_dict",
    "student", "generator", "gen", "G", "module"
]

def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    k, v = next(iter(obj.items()))
    if not isinstance(k, str):
        return False
    return isinstance(v, (torch.Tensor, Parameter))

def _flatten_once(d: dict):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat

def _maybe_strip_prefix(sd: dict, prefixes=("module.", "model.")) -> dict:
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def _find_state_dict(ckpt):
    # 1) checkpoint 자체가 state_dict인 경우 (OrderedDict 포함)
    if _looks_like_state_dict(ckpt):
        return ckpt, "<root>"

    # 2) 흔한 키들 1단계
    if isinstance(ckpt, dict):
        for k in _COMMON_KEYS:
            if k in ckpt and _looks_like_state_dict(ckpt[k]):
                return ckpt[k], k

        # 3) Lightning 스타일: {'state_dict': {...}}
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict) and _looks_like_state_dict(ckpt["state_dict"]):
            return ckpt["state_dict"], "state_dict"

        # 4) 중첩 딕셔너리 한 단계 더 탐색
        for ok, ov in ckpt.items():
            if isinstance(ov, dict):
                for k in _COMMON_KEYS:
                    if k in ov and _looks_like_state_dict(ov[k]):
                        return ov[k], f"{ok}.{k}"
                if _looks_like_state_dict(ov):  # 중첩 자체가 state_dict
                    return ov, ok

        # 5) 한 번 flat 해서 재시도 (가끔 'model.state_dict' 같은 형태)
        flat = _flatten_once(ckpt)
        for k, v in flat.items():
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v, k

    # 실패
    return None, ""

def _best_align_to_model_keys(sd: dict, model_keys: set):
    """
    원본 / 'module.' 'model.' 제거 버전 중에서
    모델 키와의 교집합이 가장 큰 버전을 선택.
    """
    candidates = [sd, _maybe_strip_prefix(sd, ("module.",)), _maybe_strip_prefix(sd, ("model.",)), _maybe_strip_prefix(sd, ("module.", "model."))]
    best = sd
    best_hits = -1
    for cand in candidates:
        hits = sum(1 for k in cand.keys() if k in model_keys)
        if hits > best_hits:
            best, best_hits = cand, hits
    return best

def _normalize_incompat(ret):
    """
    load_state_dict 반환이 PyTorch 버전에 따라 IncompatibleKeys 또는 (missing, unexpected) 튜플일 수 있음.
    이를 (list_missing, list_unexpected)로 통일.
    """
    if hasattr(ret, "missing_keys") or hasattr(ret, "unexpected_keys"):
        missing = list(getattr(ret, "missing_keys", []))
        unexpected = list(getattr(ret, "unexpected_keys", []))
        return missing, unexpected
    if isinstance(ret, (tuple, list)) and len(ret) == 2:
        return list(ret[0]), list(ret[1])
    # 알 수 없는 형식은 안전하게 빈 목록 반환
    return [], []

def _remap_for_fadnet(model, pretrained_state_dict, logger=None):
    """
    FADNet++ ckpt에서 사용하던 키 이름(예: 'dispnetc.xxx')을
    현재 FADNet 구현의 키 이름(예: 'extract_network.xxx', 'cunet.xxx')으로 매핑.

    - 구 ckpt: 'dispnetc.conv1/2/3.*'  -> 현 모델: 'extract_network.conv1/2/3.*'
    - 구 ckpt: 그 외 dispnetc.* (conv_redir, conv3_1, conv4~, iconv~, upconv~, upflow~, pred_flow~)
              -> 현 모델: 'cunet.*'
    - dispnetres.* 는 이름이 같으므로 그대로 사용 (shape이 안 맞으면 자동으로 버려짐)
    """
    # DDP 래핑 되어 있을 수도 있어서 클래스 이름만 사용
    if model.__class__.__name__ != "FADNet":
        return pretrained_state_dict

    model_state = model.state_dict()

    # dispnetc.* 키가 없으면 매핑할 게 없으니 그대로 반환
    has_dispnetc = any(k.startswith("dispnetc.") for k in pretrained_state_dict.keys())
    if not has_dispnetc:
        return pretrained_state_dict

    if logger:
        logger.info("[FADNet] Detected dispnetc.* keys in ckpt, applying key remapping to extract_network/cunet.")

    new_sd = {}

    for k, v in pretrained_state_dict.items():
        new_key = k

        if k.startswith("dispnetc."):
            # dispnetc.conv1/conv2/conv3 -> extract_network.*
            if (
                k.startswith("dispnetc.conv1.") or
                k.startswith("dispnetc.conv2.") or
                k.startswith("dispnetc.conv3.")
            ):
                suffix = k[len("dispnetc"):]  # ".conv1.0.weight" 같은 부분
                new_key = "extract_network" + suffix
            else:
                # 그 외 dispnetc.* -> cunet.*
                suffix = k[len("dispnetc"):]
                new_key = "cunet" + suffix

        elif k.startswith("dispnetres."):
            # 두 번째 블록은 이름 그대로 사용
            new_key = k

        else:
            # 기타 키들은 이름 그대로 두되, shape 맞는 것만 사용
            new_key = k

        # 실제 모델에 해당 키가 있고 shape 이 정확히 맞을 때만 채택
        if new_key in model_state and model_state[new_key].shape == v.shape:
            new_sd[new_key] = v
        else:
            # 디버깅이 필요하면 아래 주석을 풀어서 확인 가능
            if logger and new_key in model_state:
                logger.info(f"[FADNet] Skip key (shape mismatch): {k} -> {new_key}, "
                            f"ckpt={tuple(v.shape)}, model={tuple(model_state[new_key].shape)}")
            pass

    if logger:
        logger.info(f"[FADNet] Remapped {len(new_sd)}/{len(pretrained_state_dict)} ckpt params to current FADNet structure.")

    # 만약 아무 것도 매핑되지 않았다면 (실수 방지용) 원본 그대로 반환
    if not new_sd:
        if logger:
            logger.warning("[FADNet] Remap produced 0 usable params, falling back to original state_dict.")
        return pretrained_state_dict

    return new_sd
# ------------ /helpers ------------

def load_params_from_file(model, filename, device, dist_mode, logger, strict=True):
    # ---- 안전 우선 로딩 (PyTorch 2.6+: weights_only=True)
    if logger:
        logger.info(f'Loading parameters from checkpoint {filename}')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Checkpoint not found: {filename}')

    checkpoint = None
    # 1) weights_only=True (안전)
    try:
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        if logger:
            logger.info("Checkpoint loaded with weights_only=True (safe).")
    except TypeError:
        # 구버전 torch: weights_only 인자 미지원
        checkpoint = torch.load(filename, map_location=device)
        if logger:
            logger.info("Checkpoint loaded without weights_only (older torch).")
    except Exception as e_safe:
        if logger:
            logger.warning(f"Safe load failed: {e_safe}")
            logger.warning("Falling back to allowlisted globals + weights_only=False (trusted ckpt assumed).")
        # 2) 권고된 safe globals 허용 후 unsafe 로딩
        try:
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([getattr])
            checkpoint = torch.load(filename, map_location=device, weights_only=False)
            if logger:
                logger.info("Checkpoint loaded with weights_only=False (unsafe fallback).")
        except Exception as e_unsafe:
            if logger:
                logger.error(f"Fallback unsafe load also failed: {e_unsafe}")
            raise

    # 1) state_dict 찾기
    pretrained_state_dict, path = _find_state_dict(checkpoint)
    if pretrained_state_dict is None:
        keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        msg = f"[load_params_from_file] No state_dict-like content found in checkpoint. Top-level keys: {keys}"
        if logger: logger.error(msg)
        else: print(msg)
        raise KeyError("state_dict not found in checkpoint")

    # 2) DDP unwrap
    tmp_model = model.module if dist_mode else model
    state_dict = tmp_model.state_dict()
    model_keys = set(state_dict.keys())

    # 3) 접두어 정리 후 모델 키와 가장 잘 맞는 버전 선택
    pretrained_state_dict = _best_align_to_model_keys(pretrained_state_dict, model_keys)

    # 3-1) FADNet++용 ckpt라면 dispnetc.* -> extract_network./cunet. 으로 리네이밍
    pretrained_state_dict = _remap_for_fadnet(tmp_model, pretrained_state_dict, logger)

    unused_state_dict = {}
    update_state_dict = {}
    unupdate_state_dict = {}
    for key, val in pretrained_state_dict.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_state_dict[key] = val
        else:
            unused_state_dict[key] = val
    for key in state_dict:
        if key not in update_state_dict:
            unupdate_state_dict[key] = state_dict[key]

    # 4) 로드 (버전 호환)
    if strict:
        ret = tmp_model.load_state_dict(update_state_dict, strict=True)
    else:
        state_dict.update(update_state_dict)
        ret = tmp_model.load_state_dict(state_dict, strict=False)
    missing, unexpected = _normalize_incompat(ret)

    # >>> 검증 출력
    loaded_cnt = sum(p.numel() for n, p in model.named_parameters() if n not in missing)
    total_cnt  = sum(p.numel() for p in model.parameters())
    print(f"[CKPT] loaded_params={loaded_cnt}/{total_cnt}  missing={len(missing)}  unexpected={len(unexpected)}  from={path}")
    if loaded_cnt == 0:
        raise RuntimeError("[CKPT] 아무 파라미터도 로드되지 않았습니다. ckpt 경로/키 구조를 확인하세요.")

    # 가중치가 전부 0인지 간단 체크
    with torch.no_grad():
        s = 0.0
        cnt = 0
        for p in model.parameters():
            if p.numel() > 0:
                s += float(p.abs().mean().cpu())
                cnt += 1
        mean_abs = s / max(1, cnt)
        print(f"[CKPT] mean(|weights|)≈{mean_abs:.4g}")

    message = 'Unused weight: '
    for key, val in unused_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

    message = 'Not updated weight: '
    for key, val in unupdate_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)


def color_map_tensorboard(disp_gt, pred, disp_max=192):
    cm = plt.get_cmap('plasma')
    
    disp_gt = disp_gt.detach().data.cpu().numpy()
    pred = pred.detach().data.cpu().numpy()
    error_map = np.abs(pred - disp_gt)

    disp_gt = np.clip(disp_gt, a_min=0, a_max=disp_max)
    pred = np.clip(pred, a_min=0, a_max=disp_max)

    gt_tmp = 255.0 * disp_gt / disp_max
    pred_tmp = 255.0 * pred / disp_max
    error_map_tmp = 255.0 * error_map / np.max(error_map)

    gt_tmp = cm(gt_tmp.astype('uint8'))
    pred_tmp = cm(pred_tmp.astype('uint8'))
    error_map_tmp = cm(error_map_tmp.astype('uint8'))

    gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
    pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
    error_map_tmp = np.transpose(error_map_tmp[:, :, :3], (2, 0, 1))

    color_disp_c = np.concatenate((gt_tmp, pred_tmp, error_map_tmp), axis=1)
    color_disp_c = torch.from_numpy(color_disp_c)

    return color_disp_c


def write_tensorboard(tb_writer, tb_info, step):
    for k, v in tb_info.items():
        module_name = k.split('/')[0]
        writer_module = getattr(tb_writer, 'add_' + module_name)
        board_name = k.replace(module_name + "/", '')
        v = v.detach() if torch.is_tensor(v) else v
        if module_name == 'image' and v.dim() == 2:
            writer_module(board_name, v, step, dataformats='HW')
        else:
            writer_module(board_name, v, step)


def draw_depth_2_image(disp, image, baseline=0.54, focallength=1.003556e+3):
    # baseline 单位米
    disp = disp.detach().data.cpu().numpy()  # [h, w]
    image = image.detach().data.cpu().numpy()  # [3, h, w]
    image = np.transpose(image, [1, 2, 0])  # [h, w, 3]
    image = np.ascontiguousarray(image, dtype=np.uint8)

    point_size = 2
    point_colors = [(255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    thickness = -1

    depths = baseline * focallength / disp
    depths[np.isinf(depths)] = 0.0
    depths[np.isnan(depths)] = 0.

    print('depth', depths.shape)
    print('image', image.shape)
    assert depths.shape[:2] == image.shape[:2]

    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            if depths[i, j] > 40:
                point_color = point_colors[-1]
            elif depths[i, j] > 20:
                point_color = point_colors[-2]
            elif depths[i, j] > 15:
                point_color = point_colors[-3]
            elif depths[i, j] > 8:
                point_color = point_colors[-4]
            elif depths[i, j] > 0.5:
                point_color = point_colors[-5]
            else:
                continue
            cv2.circle(image, (j, i), point_size, point_color, thickness)

    image = np.transpose(image, [2, 0, 1])
    image = np.ascontiguousarray(image, dtype=np.float32)
    return torch.from_numpy(image)

def get_pos_fullres(fx, w, h):
    x_range = (np.linspace(0, w - 1, w) + 0.5 - w // 2) / fx
    y_range = (np.linspace(0, h - 1, h) + 0.5 - h // 2) / fx
    x, y = np.meshgrid(x_range, y_range)
    z = np.ones_like(x)
    pos_grid = np.stack([x, y, z], axis=0).astype(np.float32)
    return pos_grid