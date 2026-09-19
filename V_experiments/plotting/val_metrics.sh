# LightStereo
python tools/plot_val_metrics.py \
    --log train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004_datafixed/train_20260907-013005_0.log \
    --metric epe train_epe

python tools/plot_val_metrics.py \
    --log train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004_datafixed/train_20260907-013005_0.log \
    --metric loss train_loss

# LightGEV
# python tools/plot_val_metrics.py \
#     --log train_outputs/SceneFlowDataset/LightGEVModify_4/lightgev_modify_4_train_sceneflow_full_iter_6_r_075/pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF/train_20260917-161502_0.log \
#     --metric epe train_epe

# python tools/plot_val_metrics.py \
#     --log train_outputs/SceneFlowDataset/LightGEVModify_4/lightgev_modify_4_train_sceneflow_full_iter_6_r_075/pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF/train_20260917-161502_0.log \
#     --metric loss train_loss
    