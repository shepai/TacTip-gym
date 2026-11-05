"""
Visuo-tactile pipeline:

  1. Collect paired data (vision + tactile + privileged state + teacher actions):
         python collect_dataset.py --env TacArm-peg --frames 50000 --workers 4
  2. Pretrain the encoders, and see what each modality knows (R^2 table):
         python train_encoders.py --data datasets/TacArm-peg --out encoders/peg_vt.pt
     (several --data folders train one shared tactile + vision encoder across tasks)
  3. Train a policy on the encoders' outputs + an MLP (this script):
         python visuotactile_train.py

Useful comparisons, all with the same OBS_MODES:
    ENCODERS = "encoders/peg_vt.pt", FREEZE = True    pretrained, frozen
    ENCODERS = "encoders/peg_vt.pt", FREEZE = False   pretrained, fine-tuned by RL
    ENCODERS = "scratch"                              same networks, no pretraining
    ENCODERS = None                                   SB3's default CombinedExtractor
"""
from RL_BASE import evaluate, run, visualise

ENV_NAME = "TacArm-peg"
OBS_MODES = {"vision": True, "tactile": True, "privileged": False}
ENCODERS = "encoders/peg_vt.pt"
FREEZE = True
TAG = "scratch" if ENCODERS == "scratch" else ("frozen" if FREEZE else "finetuned")
MODEL_NAME = f"{ENV_NAME}_vt_{TAG}_" + "+".join(k for k, on in OBS_MODES.items() if on)
LOG_NAME = MODEL_NAME
STEPS = 3_000_000
ENV_KWARGS = {}

if __name__ == "__main__":         # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, obs_modes=OBS_MODES, env_kwargs=ENV_KWARGS,
        encoders=ENCODERS, freeze_encoders=FREEZE, record_video=True)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=50)
    visualise(ENV_NAME, MODEL_NAME)
