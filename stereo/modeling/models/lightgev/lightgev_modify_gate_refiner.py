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
# 1. Conv2d-only Geometry Encoder (C -> G), disparity 축 보존
# ---------------------------------------------------------
class GeometryEncoding1D(nn.Module):
    """
    LightStereo correlation C ∈ [B, D, H, W]를 disparity 축(D)에 대해
    1D dilated conv로 인코딩해서 geometry-aware feature G ∈ [B, K, D, H, W]를 만든다.

    [변경 1] 기존 구현은 D축을 mean(dim=2)으로 평균내 없앴다. 그러면 남는 것은
    (x, y)마다 하나의 스칼라 요약뿐이어서 봉우리 '위치' 정보가 사라지고,
    disparity별 보정을 만들 수 없다. 여기서는 D축을 끝까지 보존한다.
    같은 3-tap 커널이 모든 disparity 위치에 공유 적용되므로 D축 shift-equivariant다.
    (파라미터는 늘지 않는다 - weight sharing)
    """

    def __init__(
        self,
        in_channels: int,      # 사용하지 않음 (D는 채널이 아니라 공간축으로 취급)
        geo_channels: int = 16,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilations = dilations
        self.geo_channels = geo_channels

        branch_ch = max(geo_channels // len(dilations), 4)

        branches = []
        for d in dilations:
            pad_d = d  # k=3 기준으로 dilation=d면 pad=d -> 출력 D 유지
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
        # [변경 1] fuse도 D축을 보존한 상태에서 적용된다 (D축의 모든 위치에 동일 적용).
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
            geo_feat: [B, K, D, H, W]   (D축 보존)
        """
        B, D, H, W = cost_1_4.shape

        # [B, D, H, W] -> [B, 1, D, H*W]  (D축 = 세로, H*W = 가로)
        x = cost_1_4.view(B, 1, D, H * W)

        branch_feats = []
        for conv in self.branches:
            y = conv(x)                  # [B, branch_ch, D, H*W]  (D 유지)
            y = self.act(y)
            branch_feats.append(y)

        y = torch.cat(branch_feats, dim=1)   # [B, total_branch_ch, D, H*W]
        y = self.act(self.fuse(y))           # [B, K, D, H*W]
        return y.view(B, self.geo_channels, D, H, W)


# ---------------------------------------------------------
# 2. 작은 ConvGRU + Refiner  (변경 4~6)
# ---------------------------------------------------------
def dw_separable_conv(in_ch, out_ch, kernel_size=3):
    """depthwise 3x3 + pointwise 1x1.

    [경량화] GRU 의 3x3 conv 가 refiner 연산의 약 88% 를 차지한다.
    conv_zr(112->64, 3x3) 기준 64,512 -> 8,240 MACs/px (약 7.8배 감소).
    LightStereo 의 MobileV2Residual 과 같은 방식이라 모델 계열과도 일관된다.
    """
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=True),
    )


class MiniConvGRU(nn.Module):
    """간단한 ConvGRU (IGEV++의 ConvGRU 아이디어를 경량화한 버전)."""

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3,
                 depthwise: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        cin = hidden_dim + input_dim
        if depthwise:
            self.conv_zr = dw_separable_conv(cin, hidden_dim * 2, kernel_size)
            self.conv_q = dw_separable_conv(cin, hidden_dim, kernel_size)
        else:
            padding = kernel_size // 2
            self.conv_zr = nn.Conv2d(cin, hidden_dim * 2, kernel_size, padding=padding)
            self.conv_q = nn.Conv2d(cin, hidden_dim, kernel_size, padding=padding)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if h is None:
            b, _, h_, w_ = x.shape
            h = torch.zeros(b, self.hidden_dim, h_, w_, device=x.device, dtype=x.dtype)

        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = torch.chunk(zr, 2, dim=1)

        rq = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.conv_q(rq))

        h_new = (1.0 - z) * h + z * q
        return h_new


class LightGEVRefiner(nn.Module):
    """coarse disparity D0 를 ConvGRU 로 num_iters 회 refine.

    [변경 4] geometry volume lookup
        기존에는 D축을 평균낸 정적 geo feature 를 매 iteration 동일하게 넣었다.
        그래서 반복해도 입력이 변하지 않아, "지금 값이 맞는지" 확인할 수단이 없었고
        결국 절대 disparity 값 -> 보정량의 회귀로 퇴화했다.
        여기서는 [B,K,D,H,W] geometry volume 을 **현재 disparity 위치에서** 매 iteration
        선형보간 샘플링해서 넣는다. RAFT/IGEV 가 cross-domain 에서 버티는 핵심 장치다.

    [변경 5] disparity 정규화 + 입력 3갈래 정규화
        disp 를 raw(0~48)로 넣던 것을 max_disp/4 로 나눠 [0,1] 로 만든다.
        실측상 |d_emb|=2.59 vs |g|=0.052, |f|=0.126 로 20~50배 불균형이어서
        geometry/appearance 가 사실상 무시됐다. 세 갈래에 GroupNorm 을 걸어 균형을 맞춘다.

    [변경 6] 각 iteration 의 disparity 를 disp_list 로 반환 -> get_loss 에서 sequence loss.
    """

    def __init__(
        self,
        feat_channels: int,   # backbone 1/4 feature 채널 (ex. 24)
        geo_channels: int,    # GeometryEncoding1D 출력 채널 K
        hidden_dim: int = 32,
        num_iters: int = 2,
        disp_scale: float = 48.0,   # 1/4 해상도의 최대 disparity (= MAX_DISP // 4)
        use_lookup: bool = True,
        input_norm: bool = True,
        depthwise: bool = True,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.disp_scale = float(disp_scale)
        self.use_lookup = use_lookup

        self.geo_proj = nn.Conv2d(geo_channels, 32, 3, padding=1)
        self.feat_proj = nn.Conv2d(feat_channels, 32, 3, padding=1)

        self.disp_proj_1 = nn.Conv2d(1, 16, 7, padding=3)
        self.disp_proj_2 = nn.Conv2d(16, 16, 3, padding=1)

        # [변경 5] 세 갈래의 크기를 맞춘다
        if input_norm:
            self.geo_norm = nn.GroupNorm(8, 32)
            self.feat_norm = nn.GroupNorm(8, 32)
            self.disp_norm = nn.GroupNorm(4, 16)
        else:
            self.geo_norm = self.feat_norm = self.disp_norm = nn.Identity()

        input_dim = 32 + 32 + 16
        self.gru = MiniConvGRU(hidden_dim=hidden_dim, input_dim=input_dim,
                               kernel_size=3, depthwise=depthwise)

        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    @staticmethod
    def sample_geo_at_disp(geo_volume: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """geometry volume 을 현재 disparity 위치에서 선형보간 샘플링.

        Args:
            geo_volume: [B, K, D, H, W]
            disp:       [B, 1, H, W]  (1/4 해상도 단위)
        Returns:
            [B, K, H, W]
        """
        B, K, D, H, W = geo_volume.shape
        d = disp.clamp(0, D - 1)                       # [B,1,H,W]
        d0 = torch.floor(d)
        frac = (d - d0)                                # [B,1,H,W]
        i0 = d0.long().clamp(0, D - 1)
        i1 = (i0 + 1).clamp(0, D - 1)

        idx0 = i0.unsqueeze(1).expand(B, K, 1, H, W)
        idx1 = i1.unsqueeze(1).expand(B, K, 1, H, W)
        g0 = geo_volume.gather(2, idx0).squeeze(2)     # [B,K,H,W]
        g1 = geo_volume.gather(2, idx1).squeeze(2)
        return g0 * (1.0 - frac) + g1 * frac

    def forward(
        self,
        disp_init: torch.Tensor,    # [B,1,H,W]  D0
        feat_1_4: torch.Tensor,     # [B, C_feat, H, W]
        geo_feat_2d: torch.Tensor,  # [B, K, H, W]      lookup 미사용 시의 정적 입력
        geo_volume=None,            # [B, K, D, H, W]   lookup 사용 시
    ):
        if self.num_iters <= 0:
            return disp_init, [disp_init]

        f = self.feat_norm(F.relu(self.feat_proj(feat_1_4), inplace=True))

        use_lookup = self.use_lookup and geo_volume is not None
        if not use_lookup:
            g_static = self.geo_norm(F.relu(self.geo_proj(geo_feat_2d), inplace=True))

        disp = disp_init
        h = None
        disp_list = []

        for _ in range(self.num_iters):
            if use_lookup:
                # [변경 4] 매 iteration 현재 disparity 에서 다시 뽑는다 -> 입력이 실제로 변한다
                g_at_d = self.sample_geo_at_disp(geo_volume, disp)
                g = self.geo_norm(F.relu(self.geo_proj(g_at_d), inplace=True))
            else:
                g = g_static

            # [변경 5] 절대 disparity 대신 [0,1] 정규화 값
            d_in = disp / self.disp_scale
            d_emb = F.relu(self.disp_proj_1(d_in), inplace=True)
            d_emb = self.disp_norm(F.relu(self.disp_proj_2(d_emb), inplace=True))

            x = torch.cat([g, f, d_emb], dim=1)

            h = self.gru(h, x)
            delta_disp = self.delta_head(h)

            disp = disp + delta_disp
            disp_list.append(disp)

        return disp, disp_list


# ---------------------------------------------------------
# 3. 전체 모델: LightStereo + GEV + bounded multiplicative Geo Gate
# ---------------------------------------------------------
class LightGEVModify_2(nn.Module):
    """
    LightStereo + Lightweight Geometry Encoding Volume + 작은 ConvGRU Refiner.

    원본 LightGEV 대비 변경점 (gate 경로만):
      [변경 1] GeometryEncoding1D에서 disparity 축(D)을 보존 (mean(dim=2) 제거)
      [변경 2] geo_to_cost(Conv2d 16->48, D축 위치별 독립 가중치)를
               gate_head(Conv2d K->1, 1x1, D축 공유)로 교체
               -> D축 shift-equivariant, 파라미터 6,912 -> 17
      [변경 3] additive gate (gwc + alpha*geo_cost) 를
               bounded multiplicative gate 로 교체:
                   gwc * (1 + r * tanh(gate_head(G)))
               r = GEV_GATE_RANGE (학습하지 않는 고정 하이퍼파라미터, 기본 0.5)
               learnable geo_gate_alpha 는 제거

    변경 3의 성질:
      - r < 1 이면 곱수 (1-r, 1+r) 가 항상 양수 -> cost 부호를 뒤집을 수 없다.
        (correlation의 30~36%가 음수이므로 부호 보존이 실제로 중요하다)
      - gwc ≈ 0 (매칭 증거 없음)인 곳에서는 gate가 아무 값도 만들어내지 못한다.
        증거가 없는 영역을 채우는 것은 공간 수용영역을 가진 aggregation의 역할.
      - gate_head를 zero-init 하므로 학습 시작 시점의 모델은 정확히 LightStereo다.

    Ablation 플래그:
        - USE_GEV:           GEV 완전 ON/OFF
        - USE_GEO_COST_GATE: Geo->Cost gate ON/OFF
        - USE_GEV_REFINER:   GRU refinement(D0->DT) ON/OFF
        - GEV_ITERS:         GRU iteration 수 (0이면 update 안 함)
        - GEV_GATE_RANGE:    gate 변조 범위 r (0 < r < 1)
        - GEV_GATE_NORM:     gate_head 입력의 위치별 채널 L2 정규화 (기본 true)
        - GEV_LOOKUP:        refiner 가 매 iteration geometry volume 을 현재 disparity
                             위치에서 샘플링 (기본 true)
        - GEV_REFINER_NORM:  refiner GRU 입력 3갈래 GroupNorm (기본 true)
        - GEV_GRU_DW:        GRU conv 를 depthwise-separable 로 (기본 true)
        - GEV_SEQ_LOSS:      iteration 별 gamma 가중 sequence loss (기본 true)
        - GEV_SEQ_GAMMA:     그 gamma (기본 0.9)
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
        # Geometry encoder + Geo->Cost gate + Refiner
        # ------------------------------
        geo_channels = getattr(cfgs, "GEV_CHANNELS", 16)
        gev_hidden = getattr(cfgs, "GEV_HIDDEN_DIM", 32)
        gev_iters = getattr(cfgs, "GEV_ITERS", 2)
        dilations = getattr(cfgs, "DILATIONS", (1, 2, 4))

        # [변경 3] gate 변조 범위 r. 학습하지 않는 고정값.
        gate_range = float(getattr(cfgs, "GEV_GATE_RANGE", 0.5))
        assert 0.0 < gate_range < 1.0, (
            "GEV_GATE_RANGE must satisfy 0 < r < 1 so that the multiplier "
            "(1-r, 1+r) stays positive and the gate cannot flip the sign of "
            "the correlation volume. got {}".format(gate_range)
        )
        self.gate_range = gate_range

        # [변경 3b] gate_head 입력의 위치별 채널 L2 정규화 여부.
        # tanh 포화(-> gradient 소멸, 이진 마스크화)를 막는다. 기본 ON.
        self.gate_norm = bool(getattr(cfgs, "GEV_GATE_NORM", False))

        # GEV 사용 여부에 따라 모듈 생성
        if self.use_gev:
            self.geo_encoder = GeometryEncoding1D(
                in_channels=corr_channels,
                geo_channels=geo_channels,
                dilations=dilations,
            )
        else:
            self.geo_encoder = None

        # [변경 2] Geo->Cost gate head: K -> 1, 1x1, D축 전체에 동일 적용
        if self.use_gev and self.use_geo_cost_gate:
            self.gate_head = nn.Conv2d(geo_channels, 1, kernel_size=1, bias=True)
            # zero-init: 초기 곱수가 정확히 1.0 -> 학습 시작 모델 == LightStereo
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)
        else:
            self.gate_head = None

        # [변경 4~6] Refiner 설정
        self.use_lookup = bool(getattr(cfgs, "GEV_LOOKUP", True))
        self.seq_gamma = float(getattr(cfgs, "GEV_SEQ_GAMMA", 0.9))
        self.use_seq_loss = bool(getattr(cfgs, "GEV_SEQ_LOSS", True))
        refiner_norm = bool(getattr(cfgs, "GEV_REFINER_NORM", True))
        gru_depthwise = bool(getattr(cfgs, "GEV_GRU_DW", True))

        # GRU Refiner
        if self.use_gev and self.use_gev_refiner:
            feat_1_4_channels = self.backbone.output_channels[0]
            self.refiner = LightGEVRefiner(
                feat_channels=feat_1_4_channels,
                geo_channels=geo_channels,
                hidden_dim=gev_hidden,
                num_iters=gev_iters,
                disp_scale=self.max_disp // 4,
                use_lookup=self.use_lookup,
                input_norm=refiner_norm,
                depthwise=gru_depthwise,
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
        # GEV + bounded multiplicative Geo Gate (Ablation)
        # ------------------------------
        geo_feat = None          # [B, K, D, H4, W4]  (D축 보존)
        geo_feat_2d = None       # [B, K, H4, W4]     (refiner 입력용)
        if self.use_gev and self.geo_encoder is not None:
            geo_feat = self.geo_encoder(gwc)
            # refiner는 이번 단계에서 손대지 않으므로 2D 요약을 넘긴다.
            # (현재 disparity 위치에서 샘플링하는 cost lookup은 다음 단계)
            geo_feat_2d = geo_feat.mean(dim=2)

        gwc_enh = gwc
        if (
            self.use_gev
            and self.use_geo_cost_gate
            and geo_feat is not None
            and self.gate_head is not None
        ):
            B, K, D, H4, W4 = geo_feat.shape

            # [변경 3b] gate_head 입력을 위치별 채널 L2 정규화.
            #
            # geo_encoder 에 정규화가 전혀 없어서 geo_feat 의 크기가 자유롭게 커지고,
            # 그 결과 logit 이 커져 tanh 가 포화한다. r=0.5 로 학습한 실측:
            #   |logit| mean 4.90 / p95 19.2, 65% 가 |logit|>2 (tanh 미분 ~ 0 -> gradient 소멸)
            #   곱수가 정확히 0.5 / 1.5 에 붙어 graded modulation 이 이진 마스크로 퇴화
            # 채널 축(dim=1) 만 정규화하므로 D축 shift-equivariance 는 그대로 유지된다.
            #
            # trade-off: geo_feat 의 전체 크기(= correlation 크기에 비례, 도메인마다 변함)를
            # 버리고 채널 '패턴' 만 남긴다. 크기 자체가 도메인 의존적인 양이므로 의도된 선택이지만,
            # "매칭 신호가 전반적으로 강한가" 라는 정보는 gate 가 더 이상 보지 못한다.
            # 이 선택을 끄고 비교하려면 config 에 GEV_GATE_NORM: false 를 넣으면 된다.
            geo_in = F.normalize(geo_feat, dim=1) if self.gate_norm else geo_feat

            # [변경 2] 1x1 conv를 D축의 모든 위치에 동일하게 적용
            logit = self.gate_head(geo_in.flatten(3)).view(B, D, H4, W4)
            # [변경 3] bounded multiplicative gate: 부호 보존 + 증거 없으면 무동작
            gwc_enh = gwc * (1.0 + self.gate_range * torch.tanh(logit))

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
        # Stage 2: Geometry-aware ConvGRU refinement (D0 -> D_T)
        # ------------------------------
        if (
            self.use_gev
            and self.use_gev_refiner
            and geo_feat_2d is not None
            and self.refiner is not None
        ):
            disp_ref_low, disp_seq = self.refiner(
                disp_low, feat_1_4, geo_feat_2d,
                geo_volume=geo_feat if self.use_lookup else None)
        else:
            # Refiner OFF -> D0를 그대로 사용
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
    # Loss  (이번 단계에서는 변경 없음)
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

    # def get_loss(self, model_pred, input_data):
    #     disp_gt = input_data["disp"]  # [bz, h, w]
    #     disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
    #     mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

    #     disp_pred = model_pred['disp_pred']
    #     loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

    #     # [변경 6] coarse aux 항을 iteration 별 gamma 가중합으로 (총 가중치는 0.3 유지).
    #     # 지금까지 disp_seq_1_4 를 만들어놓고 쓰지 않아 중간 iteration 에 아무 제약이 없었다.
    #     # refiner 가 꺼져 있으면 seq 길이가 1 이라 0.3 * L(disp_4) 와 정확히 같아진다
    #     # (= LightStereo 와 동일한 loss).
    #     seq = model_pred.get('disp_seq_1_4') if self.use_seq_loss else None
    #     if seq:
    #         n = len(seq)
    #         weights = [self.seq_gamma ** (n - 1 - i) for i in range(n)]
    #         wsum = sum(weights)
    #         aux = 0.0
    #         for w, d_low in zip(weights, seq):
    #             d_up = F.interpolate(d_low, size=disp_pred.shape[-2:],
    #                                  mode='bilinear', align_corners=False) * 4.0
    #             aux = aux + (w / wsum) * F.smooth_l1_loss(
    #                 d_up[mask], disp_gt[mask], reduction='mean')
    #         loss = loss + 0.3 * aux
    #     else:
    #         disp_4 = model_pred['disp_4']
    #         loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

    #     loss_info = {'scalar/train/loss_disp': loss.item()}

    #     return loss, loss_info
