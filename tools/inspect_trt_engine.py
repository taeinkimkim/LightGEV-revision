"""빌드된 TensorRT 엔진의 레이어별 정밀도를 세어 본다.

INT8 빌드에서 "Missing scale and zero-point for tensor ..." 경고가 뜨면, 그 텐서를
쓰는 레이어는 INT8 이 아니라 FP16/FP32 로 돌아간다. 경고가 몇 줄 나오는 것 자체는
정상이지만(모든 연산에 INT8 커널이 있는 건 아니다), **대부분의 레이어가 떨어져
나갔다면 INT8 빌드를 한 의미가 없다.** 그걸 확인하는 도구다.

EngineInspector 로 엔진 안을 들여다보고 레이어별 정밀도를 집계한다.

Usage:
    python3 tools/inspect_trt_engine.py --engine a_int8.engine
    python3 tools/inspect_trt_engine.py --engine a_int8.engine a_fp16.engine   # 비교
    python3 tools/inspect_trt_engine.py --engine a_int8.engine --list-layers   # 레이어별로
"""

import argparse
import collections
import json
import os


def layer_info(engine_path, verbose=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("엔진 로드 실패: {}".format(engine_path))

    inspector = engine.create_engine_inspector()
    # DETAILED 는 레이어별 정밀도까지 담는다. 엔진이 크면 출력도 크다.
    fmt = trt.LayerInformationFormat.JSON
    try:
        inspector.execution_context = engine.create_execution_context()
    except Exception:
        pass
    raw = inspector.get_engine_information(fmt)
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw

    layers = info.get("Layers", info if isinstance(info, list) else [])
    return layers, raw


def summarize(layers):
    """레이어별 정밀도 집계. 키 이름은 TRT 버전마다 조금 다르다."""
    prec = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    named = []
    for L in layers:
        if not isinstance(L, dict):
            continue
        p = (L.get("Precision") or L.get("precision")
             or L.get("ComputePrecision") or "UNKNOWN")
        t = L.get("LayerType") or L.get("ParameterType") or "?"
        n = L.get("Name") or L.get("name") or "?"
        prec[str(p)] += 1
        by_type[str(p)][str(t)] += 1
        named.append((str(p), str(t), str(n)))
    return prec, by_type, named


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", nargs="+", required=True)
    p.add_argument("--list-layers", dest="list_layers", action="store_true",
                   help="레이어를 하나씩 찍는다 (정밀도, 타입, 이름)")
    p.add_argument("--non-int8-only", dest="non_int8_only", action="store_true",
                   help="--list-layers 와 함께: INT8 이 아닌 레이어만 찍는다")
    args = p.parse_args()

    for path in args.engine:
        if not os.path.isfile(path):
            print("[skip] 파일 없음: {}".format(path), flush=True)
            continue
        print("\n" + "=" * 70, flush=True)
        print(" {}  ({:.1f} MB)".format(os.path.basename(path),
                                        os.path.getsize(path) / 1024 ** 2), flush=True)
        print("=" * 70, flush=True)
        try:
            layers, raw = layer_info(path)
        except Exception as e:
            print("  실패: {}: {}".format(type(e).__name__, e), flush=True)
            continue
        if layers is None:
            print("  JSON 파싱 실패. 원문 앞부분:\n{}".format(raw[:400]), flush=True)
            continue

        prec, by_type, named = summarize(layers)
        total = sum(prec.values())
        if total == 0:
            print("  레이어 정보를 못 읽었다 (TRT 버전 차이). 원문 앞부분:", flush=True)
            print("  {}".format(raw[:300]), flush=True)
            continue

        print("  레이어 {} 개".format(total), flush=True)
        for pname, cnt in prec.most_common():
            print("    {:<10} {:>5} 개  ({:5.1f}%)".format(pname, cnt, 100.0 * cnt / total),
                  flush=True)

        int8_cnt = sum(c for k, c in prec.items() if "INT8" in k.upper())
        share = 100.0 * int8_cnt / total
        print("\n  INT8 비중: {:.1f}%".format(share), flush=True)
        if share < 10:
            print("  -> 사실상 INT8 이 적용되지 않았다. calibration cache 가 낡았거나"
                  " 이 구조가 INT8 커널을 거의 못 쓴다.", flush=True)
        elif share < 50:
            print("  -> 절반도 안 된다. 속도 이득이 작을 것이다. FP16 과 실측 비교할 것.", flush=True)
        else:
            print("  -> 대부분 INT8 이다. 이제 정확도를 확인할 차례다.", flush=True)

        # INT8 이 아닌 레이어가 어떤 종류인지 보면 원인이 보인다
        non_int8 = {k: v for k, v in by_type.items() if "INT8" not in k.upper()}
        if non_int8:
            print("\n  INT8 이 아닌 레이어의 종류:", flush=True)
            for pname, types in non_int8.items():
                top = ", ".join("{} x{}".format(t, c) for t, c in types.most_common(5))
                print("    {:<10} {}".format(pname, top), flush=True)

        if args.list_layers:
            print("\n  {:<10}{:<22}{}".format("precision", "type", "name"), flush=True)
            for pname, t, n in named:
                if args.non_int8_only and "INT8" in pname.upper():
                    continue
                print("  {:<10}{:<22}{}".format(pname, t[:21], n[:70]), flush=True)


if __name__ == "__main__":
    main()
