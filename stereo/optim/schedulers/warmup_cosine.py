import math
from typing import List, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine decay (no restarts).
    - warmup_steps 동안: init_lr * (warmup_factor -> 1.0) 선형 증가
    - 그 이후: min_lr까지 cosine으로 완만 하강
    - step 단위/epoch 단위 모두 지원 (last_epoch = 현재 step/epoch 인덱스)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lrs: Optional[List[float]] = None,
        warmup_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        assert total_steps > 0, "total_steps must be > 0"
        assert 0 <= warmup_steps <= total_steps, "0 <= warmup_steps <= total_steps"
        assert 0.0 < warmup_factor <= 1.0, "warmup_factor in (0,1]"

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.warmup_factor = float(warmup_factor)

        # ✅ initial_lr 누락 방지
        for pg in optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg.get("lr", 1e-4)

        self.base_lrs = [float(pg["initial_lr"]) for pg in optimizer.param_groups]

        # ✅ min_lrs 타입 보정
        if min_lrs is None:
            self.min_lrs = [1e-6 for _ in optimizer.param_groups]
        else:
            if isinstance(min_lrs, (int, float)):
                self.min_lrs = [float(min_lrs)] * len(optimizer.param_groups)
            elif isinstance(min_lrs, list):
                if len(min_lrs) != len(optimizer.param_groups):
                    self.min_lrs = [float(min_lrs[0])] * len(optimizer.param_groups)
                else:
                    self.min_lrs = [float(x) for x in min_lrs]
            else:
                raise TypeError("min_lrs must be float or List[float]")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch  # PyTorch 관례: 현재 step index (배치/에폭 선택은 사용처에서)
        step = max(0, min(step, self.total_steps))  # guard

        lrs = []
        for base_lr, min_lr in zip(self.base_lrs, self.min_lrs):
            if self.warmup_steps > 0 and step < self.warmup_steps:
                # 선형 워밍업: base_lr * (warmup_factor -> 1.0)
                w = self.warmup_factor + (1.0 - self.warmup_factor) * (step / float(self.warmup_steps))
                lr = base_lr * w
            else:
                # 코사인 감쇠: min_lr ~ base_lr 범위에서 부드럽게 하강
                if self.warmup_steps >= self.total_steps:
                    # 워밍업만으로 끝나는 극단 케이스
                    lr = base_lr
                else:
                    t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                    cos_decay = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
                    lr = min_lr + (base_lr - min_lr) * cos_decay
            lrs.append(lr)
        return lrs