### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_pretrained_sceneflow_bs12_lr0004_datafixed_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_sceneflow.yaml \
  --ckpt train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004_datafixed/ckpt/checkpoint_epoch_89.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/lightstereo \
  --device cuda:0 \
  --max_disp 192 \
  --no_npz \
  --no_vis
