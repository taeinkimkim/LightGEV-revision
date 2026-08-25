### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# No_GEV
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_no_gev \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_no_gev.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_no_gev/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Agregation_only
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_aggregation_only \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_aggregation_only.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_aggregation_only/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Refiner_only
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_refiner_only \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_refiner_only.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_refiner_only/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_iter2
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_iter_2 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_iter_2.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_2/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_iter4
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_iter_4 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_iter_4.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_4/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_iter6
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_iter_6 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_iter_6.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_6/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_channel8
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_ch_8 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_ch_8.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_ch_8/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_channel32
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_ch_32 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_ch_32.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_ch_32/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_dilation1
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_dil_1 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_dil_1.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_dil_1/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Full_dilation2
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_dil_2 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_dil_2.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_dil_2/ablation/ckpt/checkpoint_epoch_5.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0 \
  --no_npz \
  --no_vis

# Final_2
python tools/eval_single.py \
  --model_name lightgev_pretrained_sceneflow_full_final2 \
  --cfg cfgs/lightgev/lightgev_eval_sceneflow_full_iter_2.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_2/final_2/ckpt/checkpoint_epoch_26.pth \
  --output_root outputs/sceneflow_eval \
  --device cuda:0