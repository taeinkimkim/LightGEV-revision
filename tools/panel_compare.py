import argparse, sys
from pathlib import Path

sys.path.insert(0, './')
from stereo.panel.compare_engine import PanelCompareEngine
from stereo.panel.panel_builder import PanelBuilder

def main():
    ap = argparse.ArgumentParser("Multi-model comparison panel generator (PNG-only)")
    ap.add_argument("--input_root", required=True, help="eval_multi가 만든 루트 디렉토리")
    ap.add_argument("--output", required=True, help="패널을 저장할 디렉토리")
    ap.add_argument("--models", nargs="+", help="비교할 모델명 2개 (미지정시 사전순 상위 2개 사용)")
    ap.add_argument("--stems", nargs="+", help="패널에 포함할 stem 목록 (미지정시 자동 교집합)")
    ap.add_argument("--prefix", default="album_rect", help="출력 파일 접두사")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output)
    m1, m2 = args.models

    # output이 디렉토리면 그 폴더 아래로 저장, 파일이면 부모 디렉토리에 저장
    output_dir = output / f'{m1}_vs_{m2}'
    fname_prefix = args.prefix

    output_dir.mkdir(parents=True, exist_ok=True)

    engine = PanelCompareEngine(panel_builder=PanelBuilder(background="black"))
    engine.run(
        input_root=input_root,
        output_dir=output_dir,
        models=args.models,
        stems=args.stems,
        fname_prefix=fname_prefix,
    )
    engine.run_region_error_album(
        input_root=Path(input_root),
        output_dir=Path(output_dir),
        models=args.models,     
        stems=args.stems,      
        fname_prefix="album_region_error"
    )
    
    print(f"✅ Panel comparison finished. Saved under: {output_dir}")

if __name__ == "__main__":
    main()