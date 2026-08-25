from typing import Dict, Any, Tuple
import os
import torch

def _extract_name_and_input_size(batch: Dict[str, Any]) -> Tuple[str, int, int]:
    """
    FlyingThings3D 등 계층형 데이터셋에서 파일 이름과 입력 크기 추출.

    - name: 배치 dict 안의 'name' 경로 문자열 (예: ".../A/0117/left/0006.png")
    - 반환: (stem, H, W)
      stem은 상위 폴더(A) + 하위 폴더(0117) + 파일명(확장자 제거)
      예: "A_0117_0006"
    """
    name = batch.get("name", ["000000"])[0]
    path_str = str(name)

    base = os.path.basename(path_str)  # "0006.png"
    parent = os.path.basename(os.path.dirname(path_str))  # "left"
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path_str)))  # "0117"
    greatgrand = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path_str))))  # "A"

    # "left" / "right" 폴더는 생략
    if parent.lower() in ["left", "right", "image_lcam_front", "image_rcam_front", "image_2", "image_3"]:
        stem_prefix = f"{greatgrand}_{grandparent}"  # "A_0117"
    else:
        stem_prefix = f"{grandparent}_{parent}"

    stem_file = os.path.splitext(base)[0]  # "0006"
    stem = f"{stem_prefix}_{stem_file}"    # "A_0117_0006"

    # 입력 크기 추출
    left_t = batch.get("left", None)
    if not isinstance(left_t, torch.Tensor) or left_t.ndim != 4:
        raise ValueError("'left' tensor missing or wrong shape")
    _, _, H, W = left_t.shape

    return stem, H, W