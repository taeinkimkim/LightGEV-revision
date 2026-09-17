### Zero-shot Evaluation
# ------------------------------------------------------------ [SceneFlow Dataset] ----------------------------------------------------
# BANet-2D
# python tools/eval_single.py \
#   --model_name banet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/banet_2d/banet_eval_sceneflow.yaml \
#   --ckpt ckpts/banet_2d/sceneflow.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/2d_sota \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# MobileStereoNet
# python tools/eval_single.py \
#   --model_name msnet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/msnet/msnet_eval_sceneflow.yaml \
#   --ckpt ckpts/msnet/MSNet2D_SF.ckpt \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/2d_sota \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_sota_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_sceneflow.yaml \
  --ckpt ckpts/lightstereo/LightStereo-M-SceneFlow-General.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all/2d_sota \
  --device cuda:0 \
  --no_npz \
  --no_vis





# ------------------------------------------------------------ [KITTI Dataset] ------------------------------------------------------
# BANet-2D
# python tools/eval_single.py \
#   --model_name banet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/banet_2d/banet_eval_kitti.yaml \
#   --ckpt ckpts/banet_2d/sceneflow.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/2d_sota \
#   --device cuda:0

# MobileStereoNet
# python tools/eval_single.py \
#   --model_name msnet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/msnet/msnet_eval_kitti.yaml \
#   --ckpt ckpts/msnet/MSNet2D_SF.ckpt \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/2d_sota \
#   --device cuda:0

# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_sota_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_kitti.yaml \
  --ckpt ckpts/lightstereo/LightStereo-M-SceneFlow-General.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/2d_sota \
  --device cuda:0



# ------------------------------------------------------------ [Middlebury Dataset] ------------------------------------------------------
# BANet-2D
# python tools/eval_single.py \
#   --model_name banet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/banet_2d/banet_eval_middlebury.yaml \
#   --ckpt ckpts/banet_2d/sceneflow.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/2d_sota \
#   --device cuda:0 

# MobileStereoNet
# python tools/eval_single.py \
#   --model_name msnet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/msnet/msnet_eval_middlebury.yaml \
#   --ckpt ckpts/msnet/MSNet2D_SF.ckpt \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/2d_sota \
#   --device cuda:0

# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_sota_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_middlebury.yaml \
  --ckpt ckpts/lightstereo/LightStereo-M-SceneFlow-General.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/2d_sota \
  --device cuda:0 





# ------------------------------------------------------------ [ETH3D Dataset] ------------------------------------------------------
# BANet-2D
# python tools/eval_single.py \
#   --model_name banet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/banet_2d/banet_eval_eth3d.yaml \
#   --ckpt ckpts/banet_2d/sceneflow.pth \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/2d_sota \
#   --device cuda:0 

# MobileStereoNet
# python tools/eval_single.py \
#   --model_name msnet_pretrained_sceneflow_valid_GT_revised \
#   --cfg cfgs/msnet/msnet_eval_eth3d.yaml \
#   --ckpt ckpts/msnet/MSNet2D_SF.ckpt \
#   --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/2d_sota \
#   --device cuda:0 

# LightStereo
python tools/eval_single.py \
  --model_name lightstereo_sota_pretrained_sceneflow_valid_GT_revised \
  --cfg cfgs/lightstereo/lightstereo_m_eval_eth3d.yaml \
  --ckpt ckpts/lightstereo/LightStereo-M-SceneFlow-General.pth \
  --output_root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/eth3d_eval/2d_sota \
  --device cuda:0 
