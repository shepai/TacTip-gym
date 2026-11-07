"""
Teach the wheeled humanoid (RB-Y1) with TacTip hands to pick up objects of any shape,
softness, weight and surface friction, carry them to a target and put them down
(TacHumanoid-pick).

Run check_humanoid.py first. Then compare, with the same reward, objects and curriculum:

    OBS_MODES                              what it tells you
    {"privileged": True}                   teacher: knows each object's mass, friction and
                                           stiffness, how squashed it is and the grip force
    {"object": True}                       sees where the object is (a camera tracker's
                                           output) but feels nothing: grips blind
    {"object": True, "tactile": True}      sees + feels: does touch let it grip each object
                                           firmly enough without crushing it?
    {"vision": True, "tactile": True}      head camera + touch, no state at all (hardest;
                                           obs_format "dict" and a GPU)

The gap between the second and third rows is the value of touch. Each env runs ~150 steps/s
on one core, so 16 envs do 20 M steps in about 2.5 hours.

Warm start (DEMOS > 0): pick-and-place is a long sequence (reach, squeeze, lift, carry, put
down, let go), which PPO rarely finds from scratch. Before RL, the scripted picker from
tactip_humanoid.py (it uses the simulator's state) runs DEMOS episodes, with noise added to
the actions it executes (not to the ones recorded). The policy is fitted to its actions from
the observations you train with, and the value function to its returns. During RL a fading
behaviour-cloning term (BC_WEIGHT) keeps PPO from wearing the warm start away. The demos are
cached in demos/. DEMOS = 0 is pure RL.

Logs: logs/<MODEL_NAME>/progress.csv adds pick/success, pick/level (curriculum: 0 light,
grippy, square to the hands, target ahead .. 1 up to 1 kg, slippery, any angle, target to the
side), pick/squash, pick/fail_* and every reward term (reward/...), averaged per episode.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecNormalize

from RL_BASE import env_id, evaluate, load_policy, run, visualise

ENV_NAME = "TacHumanoid-pick"
OBS_MODES = {"object": True, "tactile": True}
OBS_FORMAT = "flat"                    # "dict" for vision (CNN on the camera image)
ENV_KWARGS = {"action_mode": "bimanual", "sensor_stiffness": 10.0}
STEPS = 20_000_000
N_ENVS = 16
DEMOS = 600                            # scripted episodes for the warm start (0 = pure RL)
DEMO_LEVELS = (0.0, 0.5)               # difficulty range of the demos
DEMO_NOISE = 0.3                       # action noise while recording (the labels stay clean), so
                                       # the demos also show how to recover from small mistakes
BC_WEIGHT = 1.0                        # keep the policy near the demos during RL (0 = plain PPO),
BC_DECAY_STEPS = 10_000_000            # fading over this many steps ...
BC_FLOOR = 0.1                         # ... to this fraction of BC_WEIGHT, kept to the end
MODEL_NAME = "humanoid_" + ("+".join(k for k, on in OBS_MODES.items() if on) or "blind") + ("_demo" if DEMOS else "")
LOG_NAME = MODEL_NAME

PPO_KWARGS = dict(
    learning_rate=1e-4,
    n_steps=512,                       # x N_ENVS = 8k samples per update
    batch_size=2048,
    n_epochs=5,
    gamma=0.995,                       # episodes are 1000 steps; the payoff comes at the end
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,                    # small steps: don't undo the warm start
    policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                       activation_fn=torch.nn.ELU, log_std_init=-1.0),
)


# ----------------------------------------------------------------------------- logging
class PickLogger(BaseCallback):
    """Episode averages of success, curriculum level, squash, failures and reward terms."""
    def _on_step(self):
        for info in self.locals["infos"]:
            if "reward_terms" not in info:
                continue
            self.logger.record_mean("pick/success", float(info["is_success"]))
            self.logger.record_mean("pick/level", info["level"])
            self.logger.record_mean("pick/squash", info.get("peak_squash", 0.0))
            for reason in ("object fell off the table", "object crushed"):
                key = "pick/fail_" + reason.split()[-1]
                self.logger.record_mean(key, float(info["fail_reason"] == reason))
            self.logger.record_mean(f"object/{info['object']}", float(info["is_success"]))
            for k, v in info["reward_terms"].items():
                self.logger.record_mean(f"reward/{k}", v)
        return True


# ----------------------------------------------------------------------------- warm start
def _demo_worker(args):
    """Run the scripted picker; keep the successful episodes (obs, action, reward). With
    noise, the executed action is perturbed but the recorded one is the picker's own."""
    env_kwargs, seeds, levels, noise = args
    from tactip_mujoco_gym.envs.tactip_humanoid import ScriptedPicker
    out = []
    for seed, level in zip(seeds, levels, strict=True):
        env = gym.make(env_id(ENV_NAME), **env_kwargs, level=level)
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        picker = ScriptedPicker(env)
        obs_l, A, R = [], [], []
        while True:
            a = picker().astype(np.float32)
            obs_l.append(obs)
            A.append(a)
            run_a = np.clip(a + noise * rng.standard_normal(a.shape), -1, 1) if noise else a
            obs, r, term, trunc, info = env.step(run_a)
            R.append(r)
            if term or trunc:
                break
        env.close()
        if info["is_success"]:
            out.append((obs_l, np.array(A), np.array(R, dtype=np.float32)))
    return out


def _stack(obs_list):
    if isinstance(obs_list[0], dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    return np.stack(obs_list)


def collect_demos(env_kwargs, episodes, levels=DEMO_LEVELS, noise=DEMO_NOISE, workers=None,
                  seed=12345, cache=True):
    """Successful scripted episodes as a list of (obs list, actions, rewards)."""
    modes = "+".join(k for k, on in env_kwargs.get("obs_modes", {}).items() if on) or "blind"
    path = os.path.join("demos", f"{ENV_NAME}_{modes}_{env_kwargs.get('obs_format', 'flat')}_"
                                 f"{env_kwargs.get('action_mode', 'bimanual')}_{episodes}_noise{noise}.npz")
    if cache and os.path.exists(path):
        print(f"[warm start] loading demos from {path}")
        with np.load(path, allow_pickle=True) as f:
            return list(f["demos"])
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, episodes)
    lv = rng.uniform(*levels, episodes)
    workers = workers or max(1, min(os.cpu_count() or 1, 16))
    chunks = [(env_kwargs, seeds[i::workers], lv[i::workers], noise) for i in range(workers)]
    print(f"[warm start] running the scripted picker for {episodes} episodes on {workers} processes ...")
    demos = []
    # "spawn": forking a process that already runs torch threads can hang the children
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for part in pool.map(_demo_worker, chunks):
            demos.extend(part)
    print(f"[warm start] {len(demos)} of {episodes} scripted episodes succeeded "
          f"({sum(len(d[1]) for d in demos):,} steps)")
    if cache:
        os.makedirs("demos", exist_ok=True)
        arr = np.empty(len(demos), dtype=object)
        arr[:] = demos
        np.savez_compressed(path, demos=arr)
    return demos


class ScriptedWarmStart(BaseCallback):
    """Before the first rollout: fit the policy (mean action) and value function to the
    scripted picker's demos, set VecNormalize's statistics from them, and narrow the action
    noise so PPO starts from a policy that already picks things up.

    Then, with bc_weight > 0, keep the policy near the demos while PPO improves it (as in
    DAPG): after every rollout, bc_batches gradient steps on the demos' log-likelihood,
    weighted bc_weight, fading linearly over bc_decay_steps to bc_floor x bc_weight. Without
    it, PPO's updates wear the warm start away (in a test, plain PPO took a warm start from 80%
    to 45% success in 250k steps; with the term, success stayed at 65-75% while the curriculum
    level rose, until the term had faded to nothing)."""

    def __init__(self, env_kwargs, episodes=DEMOS, epochs=40, batch_size=1024, lr=1e-3,
                 action_std=0.25, check_episodes=10, bc_weight=BC_WEIGHT,
                 bc_decay_steps=BC_DECAY_STEPS, bc_floor=BC_FLOOR, bc_batches=20):
        super().__init__()
        self.env_kwargs = env_kwargs
        self.episodes, self.epochs, self.batch_size, self.lr = episodes, epochs, batch_size, lr
        self.action_std = action_std
        self.check_episodes = check_episodes
        self.bc_weight, self.bc_decay_steps, self.bc_batches = bc_weight, bc_decay_steps, bc_batches
        self.bc_floor = bc_floor
        self._demo_obs = self._demo_act = None

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        """The fading behaviour-cloning term (before PPO's own update on this rollout)."""
        if self._demo_act is None or self.bc_weight <= 0:
            return
        fade = max(0.0, 1.0 - self.model.num_timesteps / self.bc_decay_steps)
        lam = self.bc_weight * (self.bc_floor + (1 - self.bc_floor) * fade)
        if lam <= 0:
            return
        policy, venv = self.model.policy, self.training_env
        norm = venv.normalize_obs if isinstance(venv, VecNormalize) else (lambda o: o)
        n = len(self._demo_act)
        tot = 0.0
        for _ in range(self.bc_batches):
            idx = np.random.randint(0, n, self.batch_size)
            ob = self._demo_obs
            ob = {k: v[idx] for k, v in ob.items()} if isinstance(ob, dict) else ob[idx]
            ob = obs_as_tensor(norm(ob), policy.device)
            a = torch.as_tensor(self._demo_act[idx], device=policy.device)
            dist = policy.get_distribution(ob).distribution
            # Gaussian log-likelihood of the demo action, through the mean only
            loss = lam * (((dist.mean - a) / dist.stddev.detach()) ** 2).sum(-1).mean() / 2
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
            policy.optimizer.step()
            tot += loss.item()
        self.logger.record("pick/bc_weight", lam)
        self.logger.record("pick/bc_loss", tot / self.bc_batches)

    def _on_training_start(self):
        demos = collect_demos(self.env_kwargs, self.episodes)
        if not demos:
            print("[warm start] no successful demos - starting from scratch")
            return
        venv = self.training_env
        policy = self.model.policy
        gamma = self.model.gamma
        obs = _stack([o for d in demos for o in d[0]])
        act = np.concatenate([d[1] for d in demos]).astype(np.float32)

        # normalisation statistics from the demos (they only cover the scripted states;
        # VecNormalize keeps updating during training)
        if isinstance(venv, VecNormalize):
            if isinstance(obs, dict):
                for k in venv.norm_obs_keys:
                    venv.obs_rms[k].update(obs[k])
            else:
                venv.obs_rms.update(obs)
            for _, _, rew in demos:                       # running discounted return, as VecNormalize
                g, running = 0.0, []
                for r in rew:
                    g = g * gamma + r
                    running.append(g)
                venv.ret_rms.update(np.array(running))
            obs_n = venv.normalize_obs(obs)
        else:
            obs_n = obs

        # value targets: discounted returns of the (normalised) rewards; the demos end in
        # success, so nothing to bootstrap
        targets = []
        for _, _, rew in demos:
            r = venv.normalize_reward(rew) if isinstance(venv, VecNormalize) and venv.norm_reward else rew
            g, back = 0.0, np.zeros(len(r), np.float32)
            for t in range(len(r) - 1, -1, -1):
                g = r[t] + gamma * g
                back[t] = g
            targets.append(back)
        targets = np.concatenate(targets)

        self._demo_obs, self._demo_act = obs, act           # for the fading BC term
        n = len(act)
        opt = torch.optim.Adam(policy.parameters(), lr=self.lr)
        policy.set_training_mode(True)
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            tot_pi = tot_v = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                ob = obs_as_tensor({k: v[idx] for k, v in obs_n.items()} if isinstance(obs_n, dict) else obs_n[idx],
                                   policy.device)
                a = torch.as_tensor(act[idx], device=policy.device)
                v = torch.as_tensor(targets[idx], device=policy.device)
                mean = policy.get_distribution(ob).distribution.mean
                loss_pi = ((mean - a) ** 2).mean()
                loss_v = ((policy.predict_values(ob).flatten() - v) ** 2).mean()
                opt.zero_grad()
                (loss_pi + 0.5 * loss_v).backward()
                opt.step()
                tot_pi += loss_pi.item() * len(idx)
                tot_v += loss_v.item() * len(idx)
            if epoch in (0, self.epochs - 1) or (epoch + 1) % 5 == 0:
                print(f"[warm start] epoch {epoch + 1:3d}: action MSE {tot_pi / n:.4f}, value MSE {tot_v / n:.3f}")
        with torch.no_grad():
            policy.log_std.fill_(float(np.log(self.action_std)))
        if self.check_episodes:
            self._check(venv)

    def _check(self, venv):
        """How well the warm-started policy does on its own, before any RL."""
        env = gym.make(env_id(ENV_NAME), **self.env_kwargs, level=DEMO_LEVELS[0])
        norm = venv.normalize_obs if isinstance(venv, VecNormalize) else (lambda o: o)
        wins = 0
        for ep in range(self.check_episodes):
            obs, _ = env.reset(seed=900 + ep)
            while True:
                a, _ = self.model.predict(norm(obs), deterministic=True)
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            wins += info["is_success"]
        env.close()
        print(f"[warm start] the warm-started policy succeeds in {wins}/{self.check_episodes} "
              f"episodes at level {DEMO_LEVELS[0]} before RL")


# ----------------------------------------------------------------------------- evaluation
def evaluate_objects(model_name, levels=(0.0, 1.0), n_episodes=5):
    """Success per object at each level, with how hard it squeezed soft ones."""
    from tactip_mujoco_gym.envs.tactip_humanoid import OBJECT_NAMES
    model, env, normalise, _ = load_policy(ENV_NAME, model_name)
    raw = env.unwrapped
    for level in levels:
        print(f"\n{model_name}, level {level}:")
        for name in OBJECT_NAMES:
            wins, squash, why = 0, [], {}
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=3000 + ep, options={"object": name, "level": level})
                peak = 0.0
                while True:
                    a, _ = model.predict(normalise(obs), deterministic=True)
                    obs, _, term, trunc, info = env.step(a)
                    peak = max(peak, raw._obj[4])
                    if term or trunc:
                        break
                wins += info["is_success"]
                squash.append(peak)
                if not info["is_success"]:
                    k = info["fail_reason"] or "timeout"
                    why[k] = why.get(k, 0) + 1
            print(f"  {name:<14} success {wins}/{n_episodes}, peak squash {np.mean(squash):.2f}"
                  + (f", failed: {why}" if why else ""))
    env.close()


if __name__ == "__main__":             # required: training and the demos use subprocesses
    env_kwargs = {**ENV_KWARGS, "obs_modes": OBS_MODES, "obs_format": OBS_FORMAT}
    callbacks = [PickLogger()]
    if DEMOS:
        callbacks.insert(0, ScriptedWarmStart(env_kwargs, episodes=DEMOS))
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, n_envs=N_ENVS, obs_modes=OBS_MODES,
        obs_format=OBS_FORMAT, env_kwargs=ENV_KWARGS, algo_kwargs=PPO_KWARGS,
        callbacks=callbacks, record_video=True, video_seconds=12, checkpoint_every=2_000_000)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=20)
    evaluate_objects(MODEL_NAME)
    visualise(ENV_NAME, MODEL_NAME)
