"""Grading utilities: load a submitted ``model.pt``, check the contract, and evaluate it.

This module imports only ``torch`` (plus the package env) — NOT Stable-Baselines3. The
submitted model is a self-contained TorchScript module, so the grader does not depend on how
it was trained. ``torch.jit.load`` does not unpickle arbitrary Python, so loading a student
model cannot execute arbitrary code (unlike ``torch.load`` of a pickled object).
"""

import os
import time

import numpy as np
import torch

from .contract import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    EVAL_EPISODES_CI,
    OBS_SHAPE,
    PARAM_LIMIT,
)
from .env import PandaPushPixels


def load_policy(model_path):
    """Load a TorchScript policy onto the CPU and put it in eval mode."""
    policy = torch.jit.load(model_path, map_location="cpu")
    policy.eval()
    return policy


def count_parameters(model_path):
    policy = load_policy(model_path)
    n = int(sum(p.numel() for p in policy.parameters()))
    assert n > 0, (
        "model.pt exposes 0 parameters — export with panda_push_pixels.export_model and do "
        "NOT call torch.jit.freeze (freezing hides parameters from the grader)."
    )
    return n


def _seed_offset():
    # 0 locally / in student CI; a hidden offset injected as a GitHub Secret at final grading.
    return int(os.environ.get("EVAL_SEED_OFFSET", "0"))


@torch.no_grad()
def _act(policy, obs):
    tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)   # (1, 12, 112, 112)
    action = policy(tensor).cpu().numpy().reshape(-1)
    return np.clip(action, ACTION_LOW, ACTION_HIGH)


def check_contract(model_path):
    """Validate the submitted model against the I/O + parameter contract.

    Raises ``AssertionError`` on any violation. Returns the parameter count on success.
    """
    policy = load_policy(model_path)

    n_params = int(sum(p.numel() for p in policy.parameters()))
    assert n_params > 0, (
        "model.pt exposes 0 parameters — export with panda_push_pixels.export_model and do "
        "NOT call torch.jit.freeze (freezing hides parameters from the grader)."
    )
    assert n_params <= PARAM_LIMIT, (
        f"model has {n_params:,} parameters > limit {PARAM_LIMIT:,}"
    )

    single = torch.zeros(1, *OBS_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        out = policy(single)
    assert tuple(out.shape) == (1, ACTION_DIM), (
        f"expected action shape (1, {ACTION_DIM}), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "model produced non-finite actions"

    batch = torch.zeros(8, *OBS_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        out_batch = policy(batch)
    assert tuple(out_batch.shape) == (8, ACTION_DIM), (
        f"expected batched action shape (8, {ACTION_DIM}), got {tuple(out_batch.shape)}"
    )
    return n_params


def measure_latency(model_path, n=50):
    """Average seconds per forward pass on a single observation (CPU)."""
    policy = load_policy(model_path)
    obs = torch.zeros(1, *OBS_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):           # warmup
            policy(obs)
        start = time.perf_counter()
        for _ in range(n):
            policy(obs)
    return (time.perf_counter() - start) / n


def evaluate_policy(policy, n_episodes=EVAL_EPISODES_CI, env=None, verbose=False):
    """Run ``n_episodes`` deterministic episodes with an already-loaded ``policy`` and report metrics.

    ``policy`` is any callable mapping ``obs (1, 12, 112, 112)`` float32 -> ``action (1, 7)`` — e.g. a
    module returned by ``torch.jit.load`` (grading) or ``panda_push_pixels.extract_actor`` (pre-export
    sanity check). This function never imports Stable-Baselines3.

    Success and reward are read from the env's canonical signal. The graded metric is
    ``success_rate`` — whether the cube reached the target (dwelling there, or still there right
    at the time limit) — read from the final step's ``info["is_success"]``. ``reward`` fields are
    diagnostics only (useful while training/shaping, not what the grader gates on).
    """
    own_env = env is None
    if own_env:
        env = PandaPushPixels()

    offset = _seed_offset()
    returns = []
    successes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=offset + ep)
        episode_return = 0.0
        done = False
        while not done:
            action = _act(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        successes.append(bool(info["is_success"]))
        if verbose:
            print(f"  ep {ep:3d}: return={episode_return:6.1f}  success={successes[-1]}")

    if own_env:
        env.close()

    returns = np.asarray(returns, dtype=np.float64)
    return {
        "median_reward": float(np.median(returns)),
        "mean_reward": float(returns.mean()),
        "std_reward": float(returns.std()),
        "min_reward": float(returns.min()),
        "max_reward": float(returns.max()),
        "success_rate": float(np.mean(successes)),
        "n_episodes": int(n_episodes),
        "seed_offset": offset,
        "returns": returns.tolist(),
    }


def evaluate(model_path, n_episodes=EVAL_EPISODES_CI, env=None, verbose=False):
    """Load ``model.pt`` from ``model_path`` and evaluate it — the metric the grader computes.

    Thin wrapper over :func:`evaluate_policy` that loads the TorchScript module first. Behaviour is
    identical to the previous ``evaluate`` (the CI grader path is unchanged).
    """
    return evaluate_policy(load_policy(model_path), n_episodes=n_episodes, env=env, verbose=verbose)
