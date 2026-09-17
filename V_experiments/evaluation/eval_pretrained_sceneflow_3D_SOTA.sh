### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/igev_pp/igev_pp_eval_sceneflow.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/3d_sota \
  --device cuda:0 

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/acvnet/acvnet_eval_sceneflow.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/3d_sota \
  --device cuda:0 

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/gwcnet/gwcnet_eval_sceneflow.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/3d_sota \
  --device cuda:0 







# ------------------------------------------------------------ [KITTI Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/igev_pp/igev_pp_eval_kitti.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/3d_sota \
  --device cuda:0 

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/acvnet/acvnet_eval_kitti.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/3d_sota \
  --device cuda:0 

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/gwcnet/gwcnet_eval_kitti.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/3d_sota \
  --device cuda:0 






# ------------------------------------------------------------ [Middlebury Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/igev_pp/igev_pp_eval_middlebury.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/3d_sota \
  --device cuda:0 

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/acvnet/acvnet_eval_middlebury.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/3d_sota \
  --device cuda:0 

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/gwcnet/gwcnet_eval_middlebury.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/3d_sota \
  --device cuda:0 







# ------------------------------------------------------------ [ETH3D Dataset] ------------------------------------------------------
# IGEV++
python tools/eval_single.py \
  --model_name igev_pp_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/igev_pp/igev_pp_eval_eth3d.yaml \
  --ckpt ckpts/igev_pp/sceneflow.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/3d_sota \
  --device cuda:0 

# ACVNet
python tools/eval_single.py \
  --model_name acvnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/acvnet/acvnet_eval_eth3d.yaml \
  --ckpt ckpts/acvnet/sceneflow.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/3d_sota \
  --device cuda:0 

# GwcNet
python tools/eval_single.py \
  --model_name gwcnet_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/gwcnet/gwcnet_eval_eth3d.yaml \
  --ckpt ckpts/gwcnet/checkpoint_000014.ckpt \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/3d_sota \
  --device cuda:0 