"""Export a trained Stable-Baselines3 model to a standalone TorchScript ``model.pt``.

The exported module maps an observation ``float32 (B, 12, 96, 96)`` in ``[0, 1]`` to an action
``(B, 8)`` in ``[-1, 1]``, and loads with ``torch.jit.load`` alone — no SB3 needed at grading.

Supported algorithms:
  * off-policy actors — SAC, TD3, TQC  (traces ``policy.actor(obs, deterministic=True)``),
  * on-policy         — PPO, A2C       (traces ``policy._predict(obs, deterministic=True)``).

This module does not import Stable-Baselines3; it only duck-types the model you pass in, so
importing it stays cheap for the grader.
"""

import torch

from .contract import OBS_SHAPE


class _DeterministicActor(torch.nn.Module):
    """Off-policy (SAC/TD3/TQC): the actor's deterministic (tanh-squashed) action."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        return self.actor(obs, deterministic=True)


class _DeterministicPolicy(torch.nn.Module):
    """On-policy (PPO/A2C): ``_predict`` returns a single deterministic action tensor."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy._predict(obs, deterministic=True)


def export_model(sb3_model, path="model.pt"):
    """Trace ``sb3_model``'s deterministic actor to TorchScript and save it to ``path``."""
    policy = sb3_model.policy.to("cpu").eval()
    example = torch.zeros(1, *OBS_SHAPE, dtype=torch.float32)

    actor = getattr(policy, "actor", None)
    module = _DeterministicActor(actor) if actor is not None else _DeterministicPolicy(policy)
    module.eval()

    # NOTE: do NOT torch.jit.freeze() — freezing inlines parameters into graph constants, after
    # which `.parameters()` is empty and the grader's parameter-count check cannot see the model.
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    torch.jit.save(traced, path)
    return path


def selfcheck(sb3_model, model_path="model.pt", n_steps=10, atol=1e-4):
    """Assert the traced ``model.pt`` reproduces ``sb3_model.predict(deterministic=True)``.

    Catches the most dangerous failure mode: a model trained on uint8 observations with the
    default ``normalize_images=True`` bakes a hidden ``/255`` into the traced graph and silently
    produces garbage on the grader's float ``[0, 1]`` input. Run this before committing.
    """
    import numpy as np

    from .env import PandaLiftPixels
    from .grading import load_policy

    env = PandaLiftPixels()
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
        f"Did you train with policy_kwargs=dict(normalize_images=False) on the float env?"
    )
    return max_err
