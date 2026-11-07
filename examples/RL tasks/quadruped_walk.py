"""
Teach the Go2 with TacTip feet to walk steadily whatever the ground feels like (TacQuad-walk):
follow a velocity command over tiles that change friction, softness, bounce and roughness.

Run check_quadruped.py first. Then compare, with the same reward and terrain:

    FEET      OBS_MODES               what it tells you                       speed (1 core)
    "proxy"   {"privileged": True}    upper bound: knows the ground (teacher)  ~900 steps/s
    "proxy"   {}                      blind: joints + IMU only                 ~900 steps/s
    "tactip"  {}                      blind, on the TacTip's soft feet         ~150 steps/s
    "tactip"  {"tactile": True}       touch: does feeling the ground help?     ~150 steps/s

The last two share the body and physics, so their gap is the value of the tactile input.
Runs use one process per CPU core (N_ENVS, capped at your core count); TacTip runs are about
6x slower than proxy ones, so start with proxy runs to tune the reward and PPO settings.

Logs: logs/<MODEL_NAME>/progress.csv adds terrain/level (the curriculum: 0 flat and grippy ..
1 rough, icy, soft), terrain/fell and every reward term (reward/...), averaged per episode.
"""
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from RL_BASE import evaluate, load_policy, run, visualise

ENV_NAME = "TacQuad-walk"
FEET = "tactip"                        # "tactip" | "proxy" | "rigid"
OBS_MODES = {"tactile": True}          # {} = proprioception only; add "privileged" / "vision"
ENV_KWARGS = {"feet": FEET, "sensor_stiffness": 10.0}
STEPS = 30_000_000
N_ENVS = 16
MODEL_NAME = f"quad_{FEET}_" + ("+".join(k for k, on in OBS_MODES.items() if on) or "blind")
LOG_NAME = MODEL_NAME

# Locomotion settings (legged_gym / MuJoCo Playground style): bigger batches, fewer epochs,
# a deeper ELU network, a little entropy to keep exploring gaits.
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,                      # x N_ENVS = 16k samples per update
    batch_size=4096,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.003,
    vf_coef=1.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                       activation_fn=torch.nn.ELU, log_std_init=-0.5),
)

GROUNDS = {                            # evaluation grounds (every tile the same)
    "hard":   ({"friction": 1.0, "softness": 0.0, "dampratio": 1.0}, 0.0),
    "ice":    ({"friction": 0.15, "softness": 0.0, "dampratio": 1.0}, 0.0),
    "soft":   ({"friction": 0.8, "softness": 1.0, "dampratio": 1.0}, 0.0),
    "bouncy": ({"friction": 0.8, "softness": 0.6, "dampratio": 0.3}, 0.0),
    "rough":  (None, 1.0),             # level-1 mix: rough tiles, any feel
}


class TerrainLogger(BaseCallback):
    """Episode averages of the curriculum level, falls and each reward term."""
    def _on_step(self):
        for info in self.locals["infos"]:
            if "reward_terms" in info:
                self.logger.record_mean("terrain/level", info["terrain_level"])
                self.logger.record_mean("terrain/fell", float(bool(info["fail_reason"])))
                for k, v in info["reward_terms"].items():
                    self.logger.record_mean(f"reward/{k}", v)
        return True


def evaluate_grounds(model_name, n_episodes=5, command=(0.6, 0.0, 0.0), seconds=10.0):
    """Walk straight on each ground: how often it stays up and how well it holds the speed."""
    model, env, normalise, _ = load_policy(ENV_NAME, model_name, env_kwargs={
        "command": command, "terrain_level": 0.0, "max_steps": int(seconds * 50)})
    raw = env.unwrapped
    print(f"\n{model_name}, command {command}:")
    for name, (feel, level) in GROUNDS.items():
        raw.set_terrain_level(level)
        up, speed, slip = [], [], []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=2000 + ep, options={"feel": feel} if feel else None)
            v = []
            while True:
                action, _ = model.predict(normalise(obs), deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                v.append(raw._base_state()[1][0])
                if term or trunc:
                    break
            up.append(not term)
            speed.append(np.mean(v))
            slip.append(info["reward_terms"]["feet_slip"])
        print(f"  {name:<7}: stays up {np.mean(up):4.0%}, forward speed {np.mean(speed):.2f} m/s "
              f"(asked {command[0]:.2f}), slip penalty {np.mean(slip):.3f}")
    env.close()


if __name__ == "__main__":             # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, n_envs=N_ENVS, obs_modes=OBS_MODES,
        env_kwargs=ENV_KWARGS, algo_kwargs=PPO_KWARGS, callbacks=[TerrainLogger()],
        record_video=True, video_seconds=10, checkpoint_every=2_000_000)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=20)
    evaluate_grounds(MODEL_NAME)
    visualise(ENV_NAME, MODEL_NAME)
