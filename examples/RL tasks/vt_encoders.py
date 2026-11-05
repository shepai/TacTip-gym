"""
Visuo-tactile encoders, and the policy that uses them.

    TactileEncoder         MLP on the tactip marker vector (displacement + change, in mm).
                           Carries its own input normalisation, fitted on the training data
                           and saved with the weights, so VecNormalize must leave "tactile" alone.
    VisionEncoder          small CNN on the hand-camera image: float, channels first, [0, 1]
                           (exactly what SB3 hands a policy for a uint8 image observation).
    VisuoTactileExtractor  SB3 features extractor for dict observations: tactile -> TactileEncoder,
                           vision -> VisionEncoder, everything else (proprio, privileged, events)
                           passed straight through; the results are concatenated.
    VisuoTactilePolicy     PPO policy = VisuoTactileExtractor + MLP actor and critic heads.

Pretrained weights come from train_encoders.py. Pass encoder_path=None for the same networks
trained from scratch (a useful baseline).

Note: SB3's PPO policies re-initialise every Linear / Conv2d layer of the features extractor
(orthogonal init) after building it, which would silently erase pretrained weights.
VisuoTactilePolicy reloads them after SB3 has finished building, so use it (not
"MultiInputPolicy" + extractor kwargs) with PPO. SAC does not re-initialise, so there the
extractor can be passed through policy_kwargs (see sac_policy_kwargs).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

DEFAULT_CONFIG = dict(
    latent_dim=64,              # size of each encoder's output
    tactile_hidden=256,
    vision_channels=[32, 64, 64, 64],
    vision_hidden=256,
)


# ----------------------------------------------------------------------------- encoders
class TactileEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=64, hidden=256):
        super().__init__()
        self.in_dim = int(in_dim)
        self.register_buffer("mean", torch.zeros(self.in_dim))
        self.register_buffer("std", torch.ones(self.in_dim))
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def set_input_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class VisionEncoder(nn.Module):
    def __init__(self, in_shape, latent_dim=64, channels=(32, 64, 64, 64), hidden=256):
        """in_shape = (C, H, W). Each conv halves the image size."""
        super().__init__()
        self.in_shape = tuple(int(s) for s in in_shape)
        layers, c = [], self.in_shape[0]
        for out in channels:
            layers += [nn.Conv2d(c, out, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            c = out
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *self.in_shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))

    def forward(self, x):
        return self.head(self.cnn(x))


def build_encoders(config, tactile_dim=None, vision_shape=None):
    """Encoders from a config dict (DEFAULT_CONFIG keys). Returns (tactile or None, vision or None)."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    tactile = vision = None
    if tactile_dim is not None:
        tactile = TactileEncoder(tactile_dim, cfg["latent_dim"], cfg["tactile_hidden"])
    if vision_shape is not None:
        vision = VisionEncoder(vision_shape, cfg["latent_dim"], cfg["vision_channels"], cfg["vision_hidden"])
    return tactile, vision


def load_checkpoint(path):
    """Checkpoint written by train_encoders.py (plain tensors, lists and numbers only)."""
    return torch.load(path, map_location="cpu")


def image_space_chw(space):
    """(C, H, W) shape of an image Box, and whether the observation arrives channels-last.
    SB3 already transposes uint8 images to channels-first before the policy sees them; float
    images (depth) are not transposed, so the extractor does it."""
    shape = tuple(space.shape)
    channels_last = shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4)
    return ((shape[2], shape[0], shape[1]) if channels_last else shape), channels_last


# ----------------------------------------------------------------------------- SB3 extractor
class VisuoTactileExtractor(BaseFeaturesExtractor):
    """One encoder per observation key, outputs concatenated (late fusion). The MLP heads of
    the policy (net_arch) sit on top of this."""

    def __init__(self, observation_space: spaces.Dict, encoder_path=None, encoder_config=None,
                 freeze=True):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("VisuoTactileExtractor needs dict observations (obs_format='dict')")
        ckpt = None
        if encoder_path is not None:
            if os.path.exists(encoder_path):
                ckpt = load_checkpoint(encoder_path)
            elif encoder_config is None:
                raise FileNotFoundError(f"encoder checkpoint not found: {encoder_path}")
        cfg = dict(DEFAULT_CONFIG)
        cfg.update((ckpt or {}).get("config") or encoder_config or {})

        modules, dims, channels_last = {}, {}, {}
        for key, sp in observation_space.spaces.items():
            if key == "tactile":
                modules[key], _ = build_encoders(cfg, tactile_dim=sp.shape[0])
                dims[key] = cfg["latent_dim"]
            elif key == "vision":
                chw, channels_last[key] = image_space_chw(sp)
                _, modules[key] = build_encoders(cfg, vision_shape=chw)
                dims[key] = cfg["latent_dim"]
            else:
                modules[key] = nn.Flatten()
                dims[key] = int(np.prod(sp.shape))

        super().__init__(observation_space, features_dim=int(sum(dims.values())))
        self.encoders = nn.ModuleDict(modules)
        self._channels_last = channels_last
        self._ckpt = ckpt
        self._freeze = bool(freeze)
        self.load_pretrained()

    def load_pretrained(self):
        """Copy the pretrained encoder weights in (again) and freeze them if requested.
        Safe to call more than once; VisuoTactilePolicy calls it after SB3's weight init."""
        if self._ckpt is None:
            return
        for key in ("tactile", "vision"):
            if key not in self.encoders:
                continue
            if key not in self._ckpt:
                raise KeyError(f"the encoder checkpoint has no '{key}' encoder")
            try:
                self.encoders[key].load_state_dict(self._ckpt[key])
            except RuntimeError as err:
                raise RuntimeError(
                    f"pretrained '{key}' encoder does not match this environment's '{key}' "
                    "observation (different marker count or camera size?). Re-collect the "
                    "dataset and retrain the encoders with the current settings.") from err
            for p in self.encoders[key].parameters():
                p.requires_grad_(not self._freeze)

    def forward(self, observations):
        feats = []
        for key, enc in self.encoders.items():
            x = observations[key]
            if self._channels_last.get(key):
                x = x.permute(0, 3, 1, 2)
            feats.append(enc(x))
        return torch.cat(feats, dim=1)


# ----------------------------------------------------------------------------- policies
class VisuoTactilePolicy(MultiInputActorCriticPolicy):
    """PPO / A2C policy: VisuoTactileExtractor, then MLP actor and critic heads (net_arch).

        PPO(VisuoTactilePolicy, env, policy_kwargs=dict(encoder_path="encoders/vt.pt"))
    """

    def __init__(self, observation_space, action_space, lr_schedule, encoder_path=None,
                 encoder_config=None, freeze_encoders=True, **kwargs):
        kwargs.setdefault("features_extractor_class", VisuoTactileExtractor)
        fe_kwargs = dict(kwargs.pop("features_extractor_kwargs", None) or {})
        fe_kwargs.setdefault("encoder_path", encoder_path)
        fe_kwargs.setdefault("encoder_config", encoder_config)
        fe_kwargs.setdefault("freeze", freeze_encoders)
        kwargs["features_extractor_kwargs"] = fe_kwargs
        kwargs.setdefault("net_arch", dict(pi=[256, 256], vf=[256, 256]))
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # SB3's orthogonal init has just overwritten the extractor: put the pretrained weights back
        for fe in {id(self.pi_features_extractor): self.pi_features_extractor,
                   id(self.vf_features_extractor): self.vf_features_extractor}.values():
            if isinstance(fe, VisuoTactileExtractor):
                fe.load_pretrained()


def ppo_policy_kwargs(encoder_path=None, freeze=True):
    """policy_kwargs for PPO(VisuoTactilePolicy, ...). encoder_path=None: train from scratch."""
    config = load_checkpoint(encoder_path)["config"] if encoder_path else None
    return dict(encoder_path=encoder_path, encoder_config=config,
                freeze_encoders=bool(freeze and encoder_path))


def sac_policy_kwargs(encoder_path=None, freeze=True):
    """policy_kwargs for SAC("MultiInputPolicy", ...): SAC does not re-initialise extractors."""
    config = load_checkpoint(encoder_path)["config"] if encoder_path else None
    return dict(features_extractor_class=VisuoTactileExtractor,
                features_extractor_kwargs=dict(encoder_path=encoder_path, encoder_config=config,
                                               freeze=bool(freeze and encoder_path)))
