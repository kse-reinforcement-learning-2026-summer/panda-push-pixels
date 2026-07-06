"""Export a trained Stable-Baselines3 model to a standalone TorchScript ``model.pt``.

The exported module maps an observation ``(B, 12, 112, 112)`` uint8-valued in ``[0, 255]`` to an
action ``(B, 7)`` in ``[-1, 1]``, and loads with ``torch.jit.load`` alone — no SB3 needed at grading.
The ``/255`` normalization SB3 applies (``normalize_images=True``, the default) is baked into the
traced graph, so the standalone model reproduces training behaviour on the raw uint8 observation.

Supported algorithms (the ones allowed in Project 2):
  * on-policy  — PPO, A2C        (actor lives inside ``ActorCriticPolicy``),
  * off-policy — SAC             (separate actor; forward samples, so we rebuild the mean path),
  * off-policy — TD3, DDPG       (separate actor whose ``mu`` already ends in ``tanh``).

The extracted module contains the **actor only** (no critic), so ``model.pt`` is ~half the size
of the full policy. This module does not import Stable-Baselines3; it only reads attributes off
the model you pass in, so importing it stays cheap for the grader.
"""

import torch

from .contract import OBS_SHAPE


class _OnPolicyActor(torch.nn.Module):
    """PPO / A2C: reconstruct the actor from the ActorCriticPolicy pieces.

    Path: ``pi_features_extractor -> mlp_extractor.policy_net -> action_net`` (raw action).
    """

    def __init__(self, pi_features_extractor, policy_net, action_net, obs_divisor=255.0):
        super().__init__()
        self.features_extractor = pi_features_extractor
        self.policy_net = policy_net
        self.action_net = action_net
        self.obs_divisor = obs_divisor   # 255 to replicate SB3's normalize_images=True; 1 if it was off

    def forward(self, obs):
        features = self.features_extractor(obs / self.obs_divisor)
        latent_pi = self.policy_net(features)
        return self.action_net(latent_pi)


class _SACActor(torch.nn.Module):
    """SAC: deterministic action is ``tanh(mu(latent_pi(features)))``.

    SAC's ``Actor.forward`` samples from a squashed Gaussian; to keep the traced graph clean and
    dependency-free we rebuild the deterministic mean path and squash it ourselves.
    """

    def __init__(self, sac_actor, obs_divisor=255.0):
        super().__init__()
        self.features_extractor = sac_actor.features_extractor
        self.latent_pi = sac_actor.latent_pi
        self.mu = sac_actor.mu
        self.obs_divisor = obs_divisor

    def forward(self, obs):
        features = self.features_extractor(obs / self.obs_divisor)
        latent = self.latent_pi(features)
        return torch.tanh(self.mu(latent))


class _TD3Actor(torch.nn.Module):
    """TD3 / DDPG: ``mu`` is a full ``Sequential`` that already ends in ``tanh``.

    So the deterministic action is simply ``mu(features)`` — no extra squashing.
    """

    def __init__(self, td3_actor, obs_divisor=255.0):
        super().__init__()
        self.features_extractor = td3_actor.features_extractor
        self.mu = td3_actor.mu
        self.obs_divisor = obs_divisor

    def forward(self, obs):
        features = self.features_extractor(obs / self.obs_divisor)
        return self.mu(features)


def extract_actor(sb3_model):
    """Return a standalone actor ``nn.Module`` extracted from an SB3 model.

    Works for A2C, PPO (on-policy) and DDPG, TD3, SAC (off-policy). The returned module maps a raw
    ``obs (B, 12, 112, 112)`` uint8-valued in ``[0, 255]`` to ``action (B, 7)`` in ``[-1, 1]`` and
    contains the actor only (no critic). The SB3 ``normalize_images`` ``/255`` is replicated inside
    the returned module. The algorithm is auto-detected from the model's class name.
    """
    algo = type(sb3_model).__name__.upper()
    # Do NOT change the model's device here: extract_actor must be side-effect-free so it can be
    # called mid-training (e.g. to count params) without breaking a GPU training loop. Device
    # placement for tracing is handled by export_model.
    policy = sb3_model.policy.eval()
    # Replicate SB3's image preprocessing: normalize_images=True (the default) divides uint8 obs by
    # 255 before the features extractor. Bake the same /255 into the standalone actor so the traced
    # model reproduces training behaviour on the raw uint8 observation the grader feeds it.
    obs_divisor = 255.0 if getattr(policy, "normalize_images", True) else 1.0

    if algo in ("PPO", "A2C"):
        actor = _OnPolicyActor(
            policy.pi_features_extractor,
            policy.mlp_extractor.policy_net,
            policy.action_net,
            obs_divisor,
        )
    elif algo == "SAC":
        actor = _SACActor(policy.actor, obs_divisor)
    elif algo in ("TD3", "DDPG"):
        actor = _TD3Actor(policy.actor, obs_divisor)
    else:
        raise ValueError(
            f"Unsupported algorithm: {algo}. Supported: A2C, PPO, DDPG, TD3, SAC."
        )

    actor.eval()
    return actor


def export_model(sb3_model, path="model.pt"):
    """Extract ``sb3_model``'s deterministic actor, trace it to TorchScript, save it to ``path``.

    Tracing runs on CPU (the grader is CPU-only). The actor shares parameters with the training
    policy, so we move them to CPU for tracing and restore the original device afterwards — this
    is safe to call mid-training on a GPU model.
    """
    device = sb3_model.policy.device
    actor = extract_actor(sb3_model).to("cpu")
    example = torch.zeros(1, *OBS_SHAPE, dtype=torch.float32)

    # NOTE: do NOT torch.jit.freeze() — freezing inlines parameters into graph constants, after
    # which `.parameters()` is empty and the grader's parameter-count check cannot see the model.
    with torch.no_grad():
        traced = torch.jit.trace(actor, example)
    torch.jit.save(traced, path)
    sb3_model.policy.to(device)  # restore the training device (actor shares params with policy)
    return path


def selfcheck(sb3_model, model_path="model.pt", n_steps=10, atol=1e-4):
    """Assert the traced ``model.pt`` reproduces ``sb3_model.predict(deterministic=True)``.

    Verifies the standalone actor (with its baked-in ``/255``) matches the SB3 policy on the raw
    uint8 observation — catching any normalization/preprocessing mismatch between training and the
    exported graph. Run this before committing.
    """
    import numpy as np

    from .env import PandaPushPixels
    from .grading import load_policy

    env = PandaPushPixels()
    jit_policy = load_policy(model_path)
    obs, _ = env.reset(seed=123)

    max_err = 0.0
    for _ in range(n_steps):
        sb3_action, _ = sb3_model.predict(obs, deterministic=True)
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            jit_action = jit_policy(tensor).numpy().reshape(-1)
        err = float(np.abs(np.clip(jit_action, -1, 1) - np.clip(sb3_action, -1, 1)).max())
        max_err = max(max_err, err)

        obs, _, terminated, truncated, _ = env.step(np.clip(jit_action, -1, 1))
        if terminated or truncated:
            obs, _ = env.reset(seed=123)
    env.close()

    assert max_err < atol, (
        f"traced model.pt disagrees with SB3 by {max_err:.2e} (> {atol}). "
        f"The exported actor's /255 must match the policy's normalize_images setting."
    )
    return max_err
