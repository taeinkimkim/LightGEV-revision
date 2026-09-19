"""스테레오 모델을 ONNX 로 내보낸다 (Jetson 에서 TensorRT FP16 엔진을 굽기 위한 앞단).

대상은 tools/profile_model_runtime.py 의 ENTRIES 와 같다:
    ours / lightstereo_m / banet_2d / msnet_2d / gwcnet / acvnet / igev_pp

--- 중요한 전제 ---

1. **FP32 로 내보낸다.** FP16 변환은 TensorRT 가 엔진을 구울 때 (`trtexec --fp16`)
   한다. ONNX 를 미리 half 로 만들면 TensorRT 가 쓸 수 있는 캘리브레이션 여지가
   줄고, 오히려 정확도가 더 나빠지는 경우가 많다.

2. **학습된 checkpoint 를 반드시 싣는다.** runtime 측정과 달리 배포용 export 는
   가중치 값이 결과를 좌우한다. --ckpt 를 주지 않으면 CKPTS 기본 경로를 쓰고,
   그것도 없으면 랜덤 초기화 상태로 나가므로 경고를 띄운다.

3. **입력 해상도를 고정하는 편이 좋다.** TensorRT 는 고정 shape 에서 가장 좋은
   커널을 고른다. --dynamic 을 주면 H/W 를 동적 축으로 열어 두지만, Jetson 에서
   optimization profile 을 따로 지정해야 하고 보통 더 느리다.

4. **엔진 빌드는 Jetson 에서 해야 한다.** .onnx 는 기기 독립이지만 TensorRT
   엔진은 GPU 아키텍처·TRT 버전에 묶인다. 이 스크립트는 Jetson 에서 실행할
   trtexec 명령을 같이 찍어 준다.

--- 검증 ---

내보낸 뒤 onnxruntime 으로 돌려 PyTorch 출력과 비교한다.

중요: **PyTorch 쪽 기준값을 onnxruntime 과 같은 device 에서 뽑아야 한다.** 설치된
onnxruntime 이 CPU 전용이면 PyTorch 도 CPU 로 돌린다. CUDA PyTorch 와 CPU
onnxruntime 을 비교하면 그래프가 멀쩡해도 diff 가 크게 나온다 - 이 모델들은
192 단계 cost volume 에 softmax 를 걸어 soft-argmin 을 하므로, 커널 차이로 생긴
1e-6 수준의 logit 차이가 disparity 에서 0.1~1 px 로 증폭된다. 실측:

    PyTorch(cpu)  vs ORT(cpu) : 7.3e-04   <- 그래프가 맞다는 뜻
    PyTorch(cuda) vs ORT(cpu) : 7.9e-01   <- device 차이일 뿐

그래서 3D 모델은 CPU forward 가 느려 검증에 시간이 걸린다. 급하면 --no-verify.

Usage:
    python tools/export_onnx.py                          # 전체, 544x960
    python tools/export_onnx.py --models ours banet_2d
    python tools/export_onnx.py --resolution 384x1248 --out-dir onnx_out
    python tools/export_onnx.py --models ours --ckpt ours=path/to/ckpt.pth
    python tools/export_onnx.py --no-verify --opset 16
"""

import argparse
import os
import sys
import traceback

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_model_runtime import ENTRIES, build_model
from profile_stage_runtime import parse_resolution

# label -> 학습된 checkpoint 기본 경로. runtime 측정과 달리 배포용은 가중치가 중요하다.
CKPTS = {
    "lightgev (ours)": "train_outputs/SceneFlowDataset/LightGEVModify_4/"
                       "lightgev_modify_4_train_sceneflow_full_iter_6_r_075/"
                       "pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF/"
                       "ckpt/checkpoint_epoch_89.pth",
    "lightstereo_m": "train_outputs/SceneFlowDataset/LightStereo/"
                     "lightstereo_m_train_sceneflow/"
                     "pretrain_trainBS12_valBS12_LR0.0004_datafixed/"
                     "ckpt/checkpoint_epoch_89.pth",
    "banet_2d":      "ckpts/banet_2d/sceneflow.pth",
    "msnet_2d":      "ckpts/msnet/MSNet2D_SF.ckpt",
    "gwcnet":        "ckpts/gwcnet/checkpoint_000014.ckpt",
    "acvnet":        "ckpts/acvnet/sceneflow.ckpt",
    "igev_pp":       "ckpts/igev_pp/sceneflow.pth",
}


def find_state_dict(obj, depth=0):
    """체크포인트 포맷이 제각각이라(model_state / state_dict / 평평한 dict) 찾아 들어간다."""
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        for k in ("model_state", "model_state_dict", "state_dict", "model", "net"):
            if k in obj:
                found = find_state_dict(obj[k], depth + 1)
                if found is not None:
                    return found
        if depth < 2:
            for v in obj.values():
                found = find_state_dict(v, depth + 1)
                if found is not None:
                    return found
    return None


def load_checkpoint(model, ckpt_path, log):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = find_state_dict(ck)
    if sd is None:
        raise RuntimeError("checkpoint 안에서 state_dict 를 못 찾았다: {}".format(ckpt_path))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log("    ckpt: {}  (missing {}, unexpected {})".format(
        os.path.basename(ckpt_path), len(missing), len(unexpected)))
    if len(missing) > len(sd) * 0.5:
        log("    [경고] 절반 이상이 로드되지 않았다. 키 이름이 맞는지 확인할 것.")
    return model


class StereoWrapper(nn.Module):
    """(left, right) 텐서를 받아 disparity 한 장을 돌려준다.

    원래 모델들은 dict 를 받고 dict 를 돌려주는데, ONNX 는 텐서 in / 텐서 out 이
    다루기 쉽다. TensorRT 쪽에서도 바인딩이 단순해진다.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, left, right):
        out = self.model({"left": left, "right": right})
        if isinstance(out, dict):
            out = out["disp_pred"]
        elif isinstance(out, (list, tuple)):
            out = out[-1]
        if out.dim() == 3:          # [B,H,W] -> [B,1,H,W] 로 통일
            out = out.unsqueeze(1)
        return out


def verify(onnx_path, wrapper, left, right, log, tol=1e-4):
    """PyTorch 와 onnxruntime 을 **같은 device 에서** 돌려 비교한다.

    판정은 **상대 오차**(max abs diff / 출력 최대값)로 한다. disparity 출력은
    스케일이 170~500 px 라 절대 오차 기준은 자의적이고, 실측상 정상 export 가
    8e-04 ~ 1.1e-03 사이를 오가며 고정 임계값을 넘나든다. 상대로 보면 정상은
    ~1e-06, 깨진 그래프(IGEV++)는 7.6e-01 로 다섯 자릿수가 벌어져 판정이 흔들리지 않는다.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        log("    검증 건너뜀 (onnxruntime 없음)")
        return None

    providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in providers
    ep = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
    ref_device = "cuda" if use_cuda else "cpu"

    if not use_cuda:
        log("    검증: onnxruntime 이 CPU 전용 -> PyTorch 기준값도 CPU 로 뽑는다 "
            "(3D 모델은 오래 걸릴 수 있다)")

    was = next(wrapper.parameters()).device
    wrapper.to(ref_device)
    with torch.no_grad():
        ref = wrapper(left.to(ref_device), right.to(ref_device)).cpu()
    wrapper.to(was)

    sess = ort.InferenceSession(onnx_path, providers=[ep])
    ort_out = sess.run(None, {"left": left.cpu().numpy(), "right": right.cpu().numpy()})[0]

    diff = float((torch.from_numpy(ort_out) - ref).abs().max())
    scale = max(float(ref.abs().max()), 1e-9)
    rel = diff / scale
    flag = "OK" if rel <= tol else "[경고] 허용치 초과"
    log("    검증({}): 상대 오차 {:.1e}  (max abs diff {:.3e}, 출력 최대 {:.2f})  {}".format(
        ref_device, rel, diff, scale, flag))
    return rel


def export_one(label, cfg_path, ckpt_path, h, w, args, log):
    cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    model = build_model(cfg)

    if ckpt_path and os.path.isfile(ckpt_path):
        load_checkpoint(model, ckpt_path, log)
    else:
        log("    [경고] checkpoint 없음 -> 랜덤 초기화 상태로 내보낸다. "
            "배포용이면 --ckpt 로 지정할 것.")

    device = torch.device(args.device)
    wrapper = StereoWrapper(model).to(device).eval()   # eval() 필수 (IGEV++ 의 VALID_ITERS)

    # 고정 시드. 입력이 매번 달라지면 검증 수치도 흔들려서 판정이 뒤집힌다.
    gen = torch.Generator(device=device).manual_seed(0)
    left = torch.randn(1, 3, h, w, generator=gen, device=device)
    right = torch.randn(1, 3, h, w, generator=gen, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    onnx_path = os.path.join(args.out_dir, "{}_{}x{}.onnx".format(safe, h, w))

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"left": {2: "height", 3: "width"},
                        "right": {2: "height", 3: "width"},
                        "disp": {2: "height", 3: "width"}}

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (left, right), onnx_path,
            input_names=["left", "right"], output_names=["disp"],
            opset_version=args.opset, dynamic_axes=dynamic_axes,
            do_constant_folding=True, dynamo=False,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    log("    저장: {}  ({:.1f} MB)".format(onnx_path, size_mb))

    diff = None
    if not args.no_verify:
        diff = verify(onnx_path, wrapper, left, right, log, tol=args.tol)

    del wrapper, model, left, right
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"label": label, "path": onnx_path, "size_mb": size_mb, "diff": diff}


def trtexec_hint(onnx_path, h, w, dynamic):
    name = os.path.basename(onnx_path)
    engine = name.replace(".onnx", "_fp16.engine")
    cmd = ("/usr/src/tensorrt/bin/trtexec --onnx={} --saveEngine={} --fp16".format(name, engine))
    if dynamic:
        cmd += (" \\\n    --minShapes=left:1x3x{}x{},right:1x3x{}x{}"
                " \\\n    --optShapes=left:1x3x{}x{},right:1x3x{}x{}"
                " \\\n    --maxShapes=left:1x3x{}x{},right:1x3x{}x{}").format(
            h // 2, w // 2, h // 2, w // 2, h, w, h, w, h, w, h, w)
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="내보낼 label 부분집합 (기본: 전부). 가능: "
                             + ", ".join(e[0] for e in ENTRIES))
    parser.add_argument("--cfg", nargs="+", default=None, metavar="LABEL=PATH",
                        help="label 의 cfg 경로를 덮어쓴다")
    parser.add_argument("--ckpt", nargs="+", default=None, metavar="LABEL=PATH",
                        help="label 의 checkpoint 경로를 덮어쓴다")
    parser.add_argument("--resolution", default="544x960",
                        help="HxW. TensorRT 는 고정 shape 에서 가장 빠르다 (기본 544x960)")
    parser.add_argument("--dynamic", action="store_true",
                        help="H/W 를 동적 축으로 연다 (Jetson 에서 optimization profile 필요)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset (기본 17). TensorRT 버전이 낮으면 16 이나 13 으로 낮출 것")
    parser.add_argument("--out-dir", dest="out_dir", default="onnx_export")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-verify", dest="no_verify", action="store_true",
                        help="onnxruntime 대조 검증을 건너뛴다")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="검증 허용 **상대** 오차 = max abs diff / 출력 최대값 (기본 1e-4). "
                             "정상 export 는 ~1e-06, 깨진 그래프는 1e-01 수준이라 여유가 크다.")
    args = parser.parse_args()

    def log(msg=""):
        print(msg, flush=True)

    overrides_cfg, overrides_ckpt = {}, {}
    for item in (args.cfg or []):
        k, _, v = item.partition("="); overrides_cfg[k] = v
    for item in (args.ckpt or []):
        k, _, v = item.partition("="); overrides_ckpt[k] = v

    entries = [e for e in ENTRIES if args.models is None or e[0] in args.models]
    if not entries:
        raise SystemExit("내보낼 모델이 없다. --models 값을 확인할 것.")

    h, w = parse_resolution(args.resolution)
    log("입력 해상도 : {}x{}  ({})".format(h, w, "dynamic H/W" if args.dynamic else "고정"))
    log("opset       : {}".format(args.opset))
    log("device      : {}".format(args.device))
    log("정밀도      : FP32 로 내보낸다 (FP16 변환은 trtexec --fp16 에서)")

    ok, failed = [], []
    for label, cfg_path, _group in entries:
        log("\n" + "=" * 78)
        log(" {}".format(label))
        log("=" * 78)
        cfg_path = overrides_cfg.get(label, cfg_path)
        ckpt_path = overrides_ckpt.get(label, CKPTS.get(label))
        if not os.path.isfile(cfg_path):
            log("    [실패] cfg 없음: {}".format(cfg_path))
            failed.append((label, "cfg 없음"))
            continue
        try:
            ok.append(export_one(label, cfg_path, ckpt_path, h, w, args, log))
        except Exception as e:
            log("    [실패] {}: {}".format(type(e).__name__, str(e)[:300]))
            traceback.print_exc(limit=2)
            failed.append((label, "{}: {}".format(type(e).__name__, str(e)[:80])))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # export 가 됐다고 다 쓸 수 있는 게 아니다. 검증을 통과한 것과 통과 못 한 것을
    # 분명히 갈라서 찍는다 - 검증 실패한 그래프로 TensorRT 엔진을 구우면 조용히
    # 틀린 disparity 를 내놓는다.
    verified = [r for r in ok if r["diff"] is not None and r["diff"] <= args.tol]
    unverified = [r for r in ok if r["diff"] is None]
    mismatched = [r for r in ok if r["diff"] is not None and r["diff"] > args.tol]

    log("\n" + "=" * 78)
    log(" 요약: export 성공 {} (검증 통과 {} / 검증 실패 {} / 검증 안 함 {}), export 실패 {}".format(
        len(ok), len(verified), len(mismatched), len(unverified), len(failed)))
    log("=" * 78)
    for r in verified:
        log("  OK        {:<18}{:>8.1f} MB   상대오차 {:.1e}".format(
            r["label"], r["size_mb"], r["diff"]))
    for r in unverified:
        log("  UNCHECKED {:<18}{:>8.1f} MB   검증 안 함".format(r["label"], r["size_mb"]))
    for r in mismatched:
        log("  MISMATCH  {:<18}{:>8.1f} MB   상대오차 {:.1e}  <- PyTorch 와 출력이 다르다. "
            "쓰지 말 것".format(r["label"], r["size_mb"], r["diff"]))
    for label, why in failed:
        log("  FAIL      {:<18}{}".format(label, why))

    if mismatched:
        log("\n[경고] MISMATCH 로 표시된 모델은 그래프가 PyTorch 를 재현하지 못한다.")
        log("       이 상태로 TensorRT 엔진을 구우면 오류 없이 '틀린' disparity 가 나온다.")

    usable = verified + unverified
    if usable:
        log("\n--- Jetson 에서 실행할 명령 (엔진은 반드시 기기에서 구울 것) ---")
        for r in usable:
            log("  " + trtexec_hint(r["path"], h, w, args.dynamic))


if __name__ == "__main__":
    main()


'''
python tools/export_onnx.py --resolution 544x960 --out-dir onnx_export
'''
