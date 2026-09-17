# LightStereo
python tools/plot_val_metrics.py \
    --log train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS4_valBS4_LR0.0004_EP30_datafixed/train_20260906-225150_0.log \
    --metric epe train_epe

python tools/plot_val_metrics.py \
    --log train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS4_valBS4_LR0.0004_EP30_datafixed/train_20260906-225150_0.log \
    --metric loss train_loss

# LightGEV
# python tools/plot_val_metrics.py \
#     --log train_outputs/SceneFlowDataset/LightGEVModify/lightgev_modify_train_sceneflow_full_iter_2/pretrain_trainBS4_valBS4_LR0.0004_EP30_modify/train_20260906-011456_0.log \
#     --metric epe train_epe

# python tools/plot_val_metrics.py \
#     --log train_outputs/SceneFlowDataset/LightGEVModify/lightgev_modify_train_sceneflow_full_iter_2/pretrain_trainBS4_valBS4_LR0.0004_EP30_modify/train_20260906-011456_0.log \
#     --metric loss train_loss
    