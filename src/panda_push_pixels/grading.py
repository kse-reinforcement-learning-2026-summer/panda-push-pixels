"""Grading utilities: load a submitted ``model.pt``, check the contract, and evaluate it.

This module imports only ``torch`` (plus the package env) — NOT Stable-Baselines3. The
submitted model is a self-contained TorchScript module, so the grader does not depend on how
it was trained. ``torch.jit.load`` does not unpickle arbitrary Python, so loading a student
model cannot execute arbitrary code (unlike ``torch.load`` of a pickled object).
"""

import json
import os
import re
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
    TOUCH_DISPLACEMENT,
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
    # obs is uint8 in [0, 255]; feed it as float32 (values still 0-255) -- the exported model bakes
    # in the /255 itself (SB3 normalize_images), so we must NOT normalize here.
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

    Two tiered metrics come from the same episodes:
      * ``touch_rate``   — fraction of episodes in which the agent MOVED the cube: its centre left the
        spawn position by more than ``TOUCH_DISPLACEMENT`` at any point (direction irrelevant). This is
        the 10-pt tier.
      * ``success_rate`` — fraction of episodes that SUCCEED: the cube reached the target (dwelling
        there, or still there right at the time limit), read from the final step's ``info["is_success"]``.
        This is the 15-pt tier.
    ``reward`` fields are diagnostics only (useful while training/shaping, not what the grader gates on).
    """
    own_env = env is None
    if own_env:
        env = PandaPushPixels()

    offset = _seed_offset()
    returns = []
    successes = []
    touches = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=offset + ep)
        spawn = np.asarray(info["object_position"], dtype=np.float64)
        max_displacement = 0.0
        episode_return = 0.0
        done = False
        while not done:
            action = _act(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            disp = float(np.linalg.norm(np.asarray(info["object_position"], dtype=np.float64) - spawn))
            max_displacement = max(max_displacement, disp)
            done = terminated or truncated
        returns.append(episode_return)
        successes.append(bool(info["is_success"]))
        touches.append(bool(max_displacement > TOUCH_DISPLACEMENT))
        if verbose:
            print(f"  ep {ep:3d}: return={episode_return:6.1f}  moved={max_displacement:.3f}m  "
                  f"touch={touches[-1]}  success={successes[-1]}")

    if own_env:
        env.close()

    returns = np.asarray(returns, dtype=np.float64)
    return {
        "median_reward": float(np.median(returns)),
        "mean_reward": float(returns.mean()),
        "std_reward": float(returns.std()),
        "min_reward": float(returns.min()),
        "max_reward": float(returns.max()),
        "touch_rate": float(np.mean(touches)),
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


def notebook_trains_sb3(notebook_path):
    """5-pt training gate: does the STUDENT's own code train an allowed SB3 agent?

    Returns ``(ok, detail)``. ``ok`` is True iff the student's code (see below) both imports an allowed
    SB3 algorithm (:data:`~panda_push_pixels.contract.ALLOWED_SB3_ALGOS`) and calls ``.learn(...)``.

    Only code from the FIRST student section onward is scanned — i.e. from the first top-level markdown
    header ``## N`` with ``N >= 3``. The template's prewritten Sections 1-2 (environment demo + the naive
    PPO baseline) come before that, so the shipped baseline's import / ``.learn()`` never counts. Comments
    are stripped and markdown is ignored, so a mention of PPO in prose or a commented-out call does not
    satisfy the gate. Pure JSON parsing — this does not import or execute the notebook.
    """
    from .contract import ALLOWED_SB3_ALGOS

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])

    boundary = len(cells)                       # start of student work (first '## N', N >= 3)
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            m = re.match(r"\s*#{1,2}\s+(\d+)[.)]", "".join(c.get("source", [])))
            if m and int(m.group(1)) >= 3:
                boundary = i
                break

    code_lines = []
    for c in cells[boundary:]:
        if c.get("cell_type") != "code":
            continue
        for line in "".join(c.get("source", [])).splitlines():
            code_lines.append(line.split("#", 1)[0])    # drop trailing comments
    code = "\n".join(code_lines)

    has_import = re.search(r"\b(from|import)\s+stable_baselines3", code) is not None
    algo = next((a for a in ALLOWED_SB3_ALGOS if re.search(rf"\b{a}\b", code)), None)
    has_learn = re.search(r"\.learn\s*\(", code) is not None
    ok = has_import and algo is not None and has_learn
    detail = (f"stable_baselines3 import={has_import}, allowed algo={algo}, .learn() call={has_learn} "
              f"(scanned {sum(1 for c in cells[boundary:] if c.get('cell_type') == 'code')} student "
              f"code cell(s) from the first section >= 3)")
    return ok, detail
