from RL_BASE import*

main(ENV_NAME = "TacArm-hill",MODEL_NAME = "ppo_incline",TOTAL_TIMESTEPS = 500_000)
import os
# Force PyTorch calculation tracking onto CPU to avoid kernel image compile bugs
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

show(ENV_NAME = "TacArm-hill",MODEL_NAME = "ppo_incline",record=1,filename="/home/dexter/Documents/GitHub/TacTip-gym/assets/incline.mp4",)
#view(ENV_NAME = "TacArm-hill")