import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation


# ---------------------------------------------------------
# 1. Conv2d-only Geometry Encoder (C → G)
# ---------------------------------------------------------
class GeometryEncoding1D(nn.Module):
    """
    Conv2d만 이용해서 LightStereo correlation C ∈ [B, D, H, W]를
    disparity 축(D)에 대해 1D conv로 인코딩한 뒤,
    geometry-aware feature G ∈ [B, K, H, W]를 만드는 블록.
    """

    def __init__(
        self,
        in_channels: int,      # D (correlation 채널 수)
        geo_channels: int = 16,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilations = dilations

        branch_ch = max(geo_channels // len(dilations), 4)

        branches = []
        for d in dilations:
            pad_d = d  # k=3 기준으로 dilation=d면 pad=d
            conv = nn.Conv2d(
                in_channels=1,             # [B,1,D,H*W]
                out_channels=branch_ch,
                kernel_size=(kernel_size, 1),
                dilation=(d, 1),
                padding=(pad_d, 0),
                bias=False,
            )
            branches.append(conv)
        self.branches = nn.ModuleList(branches)

        total_branch_ch = branch_ch * len(dilations)
        self.fuse = nn.Conv2d(
            in_channels=total_branch_ch,
            out_channels=geo_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, cost_1_4: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_1_4: [B, D, H, W]  (LightStereo correlation volume)

        Returns:
            geo_feat: [B, K, H, W]
        """
        B, D, H, W = cost_1_4.shape

        # [B, D, H, W] → [B, 1, D, H*W]  (D축 = 세로, H*W = 가로)
        x = cost_1_4.view(B, 1, D, H * W)

        branch_feats = []
        for conv in self.branches:
            y = conv(x)              # [B, branch_ch, D', H*W]
            y = self.act(y)
            # D' 축 평균 → [B, branch_ch, H*W]
            y = y.mean(dim=2)
            # 다시 [B, branch_ch, H, W]
            y = y.view(B, -1, H, W)
            branch_feats.append(y)

        y = torch.cat(branch_feats, dim=1)   # [B, total_branch_ch, H, W]
        y = self.act(self.fuse(y))           # [B, geo_channels, H, W]
        return y


# ---------------------------------------------------------
# 2. 작은 ConvGRU + Refiner
# ---------------------------------------------------------
class MiniConvGRU(nn.Module):
    """
    간단한 ConvGRU (IGEV++의 ConvGRU 아이디어를 경량화한 버전).
    """

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(hidden_dim + input_dim, hidden_dim * 2,
                                 kernel_size, padding=padding)
        self.conv_q = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                kernel_size, padding=padding)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, hidden_dim, H, W]
            x: [B, input_dim, H, W]
        """
        if h is None:
            b, _, h_, w_ = x.shape
            h = torch.zeros(b, self.conv_q.out_channels, h_, w_,
                            device=x.device, dtype=x.dtype)

        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = torch.chunk(zr, 2, dim=1)

        rq = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.conv_q(rq))

        h_new = (1.0 - z) * h + z * q
        return h_new


class LightGEVRefiner(nn.Module):
    """
    LightStereo coarse disparity D0를
    geometry feature G(x,y) + backbone feature F_L(x,y)를 사용한
    작은 ConvGRU로 num_iters회 refine.
    """

    def __init__(
        self,
        feat_channels: int,   # backbone 1/4 feature 채널 (ex. 24)
        geo_channels: int,    # GeometryEncoding1D 출력 채널 K
        hidden_dim: int = 32,
        num_iters: int = 2,
    ):
        super().__init__()
        self.num_iters = num_iters

        # 각 입력을 작은 채널로 줄이는 embedding
        self.geo_proj = nn.Conv2d(geo_channels, 32, 3, padding=1)
        self.feat_proj = nn.Conv2d(feat_channels, 32, 3, padding=1)

        self.disp_proj_1 = nn.Conv2d(1, 16, 7, padding=3)
        self.disp_proj_2 = nn.Conv2d(16, 16, 3, padding=1)

        # GRU 입력 차원 = 32(geo) + 32(feat) + 16(disp_emb)
        input_dim = 32 + 32 + 16
        self.gru = MiniConvGRU(hidden_dim=hidden_dim, input_dim=input_dim, kernel_size=3)

        # disparity residual head
        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(
        self,
        disp_init: torch.Tensor,   # [B,1,H,W]  D0 (LightStereo coarse)
        feat_1_4: torch.Tensor,    # [B, C_feat, H, W]  backbone 1/4 feature
        geo_feat: torch.Tensor,    # [B, K, H, W]  geometry feature
    ):
        # iter = 0인 경우: 그냥 D0 반환 (ablation용)
        if self.num_iters <= 0:
            return disp_init, [disp_init]

        g = F.relu(self.geo_proj(geo_feat), inplace=True)
        f = F.relu(self.feat_proj(feat_1_4), inplace=True)

        disp = disp_init
        h = None
        disp_list = []

        for _ in range(self.num_iters):
            d_emb = F.relu(self.disp_proj_1(disp), inplace=True)
            d_emb = F.relu(self.disp_proj_2(d_emb), inplace=True)

            x = torch.cat([g, f, d_emb], dim=1)  # [B, input_dim, H, W]

            h = self.gru(h, x)
            delta_disp = self.delta_head(h)

            disp = disp + delta_disp
            disp_list.append(disp)

        return disp, disp_list


# ---------------------------------------------------------
# 3. 전체 모델: LightStereo + GEV + Geo-Cost Gate (Ablation 플래그 포함)
# ---------------------------------------------------------
class LightGEV(nn.Module):
    """
    LightStereo + Lightweight Geometry Encoding Volume + 작은 ConvGRU Refiner.

    Ablation 플래그:
        - USE_GEV:           GEV 완전 ON/OFF
        - USE_GEO_COST_GATE: Geo→Cost gate (C̃ = C + α φ(G)) ON/OFF
        - USE_GEV_REFINER:   GRU refinement(D0→DT) ON/OFF
        - GEV_ITERS:         GRU iteration 수 (0이면 update 안 함)
    """

    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # ---------- Ablation flags ----------
        self.use_gev = getattr(cfgs, "USE_GEV", True)
        self.use_geo_cost_gate = getattr(cfgs, "USE_GEO_COST_GATE", True)
        self.use_gev_refiner = getattr(cfgs, "USE_GEV_REFINER", True)

        # ------------------------------
        # LightStereo base (그대로 재사용)
        # ------------------------------
        self.backbone = Backbone(cfgs.get("BACKBONE", "MobileNetv2"))

        # LightStereo에서 correlation volume 채널 수 (기존 코드에서 48 사용)
        corr_channels = 48

        self.cost_agg = Aggregation(
            in_channels=corr_channels,
            left_att=self.left_att,
            blocks=cfgs.AGGREGATION_BLOCKS,
            expanse_ratio=cfgs.EXPANSE_RATIO,
            backbone_channels=self.backbone.output_channels,
        )

        # ------------------------------
        # Geometry encoder + Geo→Cost gate + Refiner
        # ------------------------------
        geo_channels = getattr(cfgs, "GEV_CHANNELS", 16)
        gev_hidden = getattr(cfgs, "GEV_HIDDEN_DIM", 32)
        gev_iters = getattr(cfgs, "GEV_ITERS", 2)
        dilations = getattr(cfgs, "DILATIONS", (1, 2, 4))

        # GEV 사용 여부에 따라 모듈 생성
        if self.use_gev:
            self.geo_encoder = GeometryEncoding1D(
                in_channels=corr_channels,
                geo_channels=geo_channels,
                dilations=dilations,
            )
        else:
            self.geo_encoder = None

        # Geo→Cost gate
        if self.use_gev and self.use_geo_cost_gate:
            self.geo_to_cost = nn.Conv2d(
                geo_channels, corr_channels, kernel_size=3, padding=1, bias=False
            )
            # geometry gate의 세기를 조절하는 learnable scalar
            self.geo_gate_alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.geo_to_cost = None
            self.geo_gate_alpha = None

        # GRU Refiner
        if self.use_gev and self.use_gev_refiner:
            feat_1_4_channels = self.backbone.output_channels[0]
            self.refiner = LightGEVRefiner(
                feat_channels=feat_1_4_channels,
                geo_channels=geo_channels,
                hidden_dim=gev_hidden,
                num_iters=gev_iters,
            )
        else:
            self.refiner = None

        # ------------------------------
        # LightStereo-style upsampling
        # ------------------------------
        self.refine_1 = nn.Sequential(
            BasicConv2d(
                self.backbone.output_channels[0], 24, 3, 1, 1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                24, 24, 3, 1, 1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.ReLU,
            ),
        )

        self.stem_2 = nn.Sequential(
            BasicConv2d(
                3, 16, 3, 2, 1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                16, 16, 3, 1, 1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
            ),
        )
        self.refine_2 = FPNLayer(24, 16)
        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

        self.use_context_up = getattr(cfgs, "USE_CONTEXT_UP", True)

    @staticmethod
    def compute_gradient(disp: torch.Tensor):
        gx = disp[..., :, 1:] - disp[..., :, :-1]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = disp[..., 1:, :] - disp[..., :-1, :]
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx, gy

    def forward(self, data):
        left = data["left"]
        right = data["right"]

        # ------------------------------
        # Stage 1: Backbone + Corr
        # ------------------------------
        feats_left = self.backbone(left)
        feats_right = self.backbone(right)

        gwc = correlation_volume(
            feats_left[0], feats_right[0], self.max_disp // 4
        )  # [B, C_corr, H4, W4]

        # ------------------------------
        # GEV + Geo-Cost Gate (Ablation)
        # ------------------------------
        geo_feat = None
        if self.use_gev and self.geo_encoder is not None:
            geo_feat = self.geo_encoder(gwc)  # [B, K, H4, W4]

        gwc_enh = gwc
        if (
            self.use_gev
            and self.use_geo_cost_gate
            and geo_feat is not None
            and self.geo_to_cost is not None
        ):
            geo_cost = self.geo_to_cost(geo_feat)          # [B, C_corr, H4, W4]
            alpha = self.geo_gate_alpha if self.geo_gate_alpha is not None else 1.0
            gwc_enh = gwc + alpha * geo_cost               # geometry-aware cost

        # LightStereo-style Aggregation
        enc = self.cost_agg(gwc_enh, feats_left)
        cost_agg = enc[0]

        squeezed = cost_agg.reshape(
            cost_agg.size(0), -1, cost_agg.size(2), cost_agg.size(3)
        )
        prob = F.softmax(squeezed, dim=1)
        disp_init_low = disparity_regression(prob, self.max_disp // 4)

        if disp_init_low.dim() == 3:
            disp_low = disp_init_low.unsqueeze(1)
        else:
            disp_low = disp_init_low  # [B,1,H4,W4]

        result = {"disp_init_1_4": disp_low}
        feat_1_4 = feats_left[0]

        # ------------------------------
        # Stage 2: Geometry-aware ConvGRU refinement (D0 → D_T)
        # ------------------------------
        if (
            self.use_gev
            and self.use_gev_refiner
            and geo_feat is not None
            and self.refiner is not None
        ):
            disp_ref_low, disp_seq = self.refiner(disp_low, feat_1_4, geo_feat)
        else:
            # Refiner OFF → D0를 그대로 사용
            disp_ref_low = disp_low
            disp_seq = [disp_low]

        result["disp_ref_low"] = disp_ref_low
        result["disp_seq_1_4"] = disp_seq  # iterative 결과들

        # ------------------------------
        # Stage 3: LightStereo context-aware upsampling
        # ------------------------------
        xspx = self.refine_1(feat_1_4)
        xspx = self.refine_2(xspx, self.stem_2(left))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, dim=1)

        if self.use_context_up:
            disp_pred = context_upsample(
                disp_ref_low * 4.0, spx_pred.float()
            ).unsqueeze(1)
        else:
            disp_up = F.interpolate(
                disp_ref_low,
                size=left.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disp_pred = disp_up * 4.0

        result["disp_pred"] = disp_pred

        if self.training:
            disp_4 = F.interpolate(
                disp_ref_low,
                size=left.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disp_4 = disp_4 * 4.0
            result["disp_4"] = disp_4

        return result

    # ------------------------------
    # Loss
    # ------------------------------
    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]
        left = input_data["left"]

        if disp_gt.dim() == 3:
            disp_gt = disp_gt.unsqueeze(1)

        mask = (disp_gt > 0) & (disp_gt < self.max_disp)

        disp_pred = model_pred["disp_pred"]  # full-resolution

        loss = F.smooth_l1_loss(
            disp_pred[mask], disp_gt[mask], reduction="mean"
        )

        # coarse 1/4 aux loss
        if "disp_4" in model_pred:
            loss += 0.3 * F.smooth_l1_loss(
                model_pred["disp_4"][mask], disp_gt[mask], reduction="mean"
            )

        # 초기 D0에 약한 supervision
        if "disp_init_1_4" in model_pred:
            disp_init_up = F.interpolate(
                model_pred["disp_init_1_4"],
                size=left.shape[2:],
                mode="bilinear",
                align_corners=False,
            ) * 4.0
            loss += 0.1 * F.smooth_l1_loss(
                disp_init_up[mask], disp_gt[mask], reduction="mean"
            )

        # plane gradient consistency
        gx_pred, gy_pred = self.compute_gradient(disp_pred)
        gx_gt, gy_gt = self.compute_gradient(disp_gt)

        loss_plane = (
            F.smooth_l1_loss(gx_pred[mask], gx_gt[mask], reduction="mean")
            + F.smooth_l1_loss(gy_pred[mask], gy_gt[mask], reduction="mean")
        ) * 0.02

        loss += loss_plane

        loss_info = {
            "scalar/train/loss_total": float(loss.item()),
            "scalar/train/loss_plane": float(loss_plane.item()),
        }

        return loss, loss_info