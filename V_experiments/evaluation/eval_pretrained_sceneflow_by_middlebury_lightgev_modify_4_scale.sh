# =================================================================================
# LightGEVModify_4 : Middlebury 해상도 sweep
#
#   모델 3개 : full_iter_6_r_075 / _dil_136 / _dil_148   (모두 epoch 89)
#   배율 3개 : 원본의 1.0 / 0.5 / 0.25 배
#
# 배율은 cfg 의 DATA_CONFIG.DATA_INFOS[0].SCALE 로 들어간다 (MiddleburyDataset).
# 이미지와 disparity 값을 같은 배율로 줄이므로 GT 최대 disparity 도 같이 줄어든다:
#
#   배율   GT 최대   MAX_DISP(192) 초과 픽셀   초과 scene
#   1.0    1074.5    23.8%                     22/23
#   0.5     537.1     4.3%                      7/23
#   0.25    268.6     0.1%                      1/23  (Shopvac 뿐)
#
# 그래서 --gt_out_of_range 를 반드시 같이 본다.
#   invalid : 범위 밖 GT 를 무효로 빼고 평가  <- 기본값으로 쓴다
#   clip    : 192 로 잘라서 평가에 포함 (기존 동작)
# clip 은 예측도 192 에서 잘리므로 범위 밖 픽셀의 오차가 0 이 되어, 1.0 배에서
# 성능이 실제보다 좋아 보인다. 배율 간 비교에는 invalid 만 쓸 것.
#
# 주의: cross-domain 수치는 run 간 편차가 크다. 모델 간 차이가 편차보다 작으면
#       단일 run 으로 순위를 매기지 말 것.
# =================================================================================

set -u

CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify_4
EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/lightgev_modify_4_scale
EPOCH=89
DEVICE=cuda:0

# 둘 다 돌린다. 집계기가 gtoor 별로 표를 따로 찍고, run 디렉토리 이름에
# _gtoor_<값>_ 이 들어가므로 서로 덮어쓰지 않는다.
GT_OORS="invalid clip"

VARIANTS="full_iter_6_r_075 full_iter_6_r_075_dil_136 full_iter_6_r_075_dil_148"
SCALES="100 050 025"

evaluate () {           # $1 = variant, $2 = scale tag (100|050|025), $3 = gt_out_of_range
    local variant=$1
    local scale=$2
    local GT_OOR=$3
    local cfg=cfgs/lightgev_modify_4/lightgev_modify_4_eval_middlebury_${variant}_scale_${scale}.yaml
    local ckpt=${CKPT_ROOT}/lightgev_modify_4_train_sceneflow_${variant}/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth

    if [ ! -f "${cfg}" ];  then echo "[skip] cfg 없음:  ${cfg}";  return; fi
    if [ ! -f "${ckpt}" ]; then echo "[skip] ckpt 없음: ${ckpt}"; return; fi

    echo "=============================================================="
    echo " ${variant}  |  scale 0.${scale}  |  gt_out_of_range=${GT_OOR}"
    echo "=============================================================="

    python tools/eval_single.py \
      --model_name lightgev_modify_4_pretrained_sceneflow_${variant}_scale_${scale}_gtoor_${GT_OOR}_bs12_lr0004_normT_biasF_valid_GT_revised \
      --cfg ${cfg} \
      --ckpt ${ckpt} \
      --output_root ${OUT_ROOT} \
      --gt_out_of_range ${GT_OOR} \
      --device ${DEVICE}
}

# 배율을 가장 바깥 루프로 둔다. 1.0 배가 가장 무거우므로(프레임당 peak ~3.8GB,
# 2020x3000) 먼저 돌려서 자원 문제를 일찍 발견한다.
# 총 3 배율 x 3 모델 x 2 gtoor = 18 run.
for SCALE in ${SCALES}; do
    for VARIANT in ${VARIANTS}; do
        for GT_OOR in ${GT_OORS}; do
            evaluate ${VARIANT} ${SCALE} ${GT_OOR}
        done
    done
done

# 표로 모으기 (gtoor 별로 표가 하나씩 나온다)
python tools/collect_middlebury_scale.py --root ${OUT_ROOT} --csv
