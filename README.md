# panda-lift-pixels

Frozen pixel-observation **Panda Lift** environment and autograder for **KSE Reinforcement
Learning — Coding Project 2**.

This package is the single source of truth for the project contract. It is installed from a
pinned git tag, so the environment and grader are byte-identical on the student's Colab, in
their GitHub Actions CI, and in the instructor's final grading run.

## The task

Grasp the cube and lift it above the table, then keep it from falling — observed **only from
pixels** (4 stacked grayscale frames, DQN-style), controlled at the **joint** level.

* **Observation** — `Box(0, 1, (4, 96, 96), float32)`: 4 stacked **grayscale** frames (96×96),
  channels-first, already normalized to `[0, 1]`. (Do **not** normalize again in your model.)
  **Why grayscale:** Scene is mostly gray (robot, table, walls); only the cube is green.
  Grayscale reduces observation 3× (faster training, fewer params) while preserving task info.
* **Action** — `Box(-1, 1, (8,), float32)`: 7 joint position deltas + gripper.
* **Canonical reward** — `0.0` while the cube is lifted-and-held, else `-1.0`, over a fixed
  50-step horizon. The graded metric is the **median cumulative reward**.

## Install

```bash
# Grading / evaluation only (CI, local tests) — no Stable-Baselines3:
pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "panda-lift-pixels @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-lift-pixels.git@v2.0.0"

# Training (Colab/Kaggle) — keep the platform's GPU torch, add the SB3 stack:
pip install "panda-lift-pixels[train] @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-lift-pixels.git@v2.0.0"
```

Requires **Python 3.11+** (panda-gym pins `numpy<2`; pybullet builds from source on 3.13+).
On Colab (3.12) pybullet builds from source the first time (~3–6 min); install `numpy<2` first.

## Public API

```python
import gymnasium as gym
import panda_lift_pixels
from panda_lift_pixels import (
    make_eval_env,      # the exact frozen env the grader uses
    export_model,       # SB3 model  ->  standalone TorchScript model.pt
    selfcheck,          # assert the exported model.pt == sb3_model.predict(deterministic=True)
    grading,            # grading.evaluate / check_contract / count_parameters / measure_latency
    contract,           # all contract constants (shapes, limits, threshold)
)

# Use either gym.make or make_eval_env
env = gym.make("PandaLiftPixels-v0")
# env = make_eval_env()  # equivalent
```

**For training:** wrap the env with your own reward shaping / curriculum logic. The frozen env
exposed `info["object_position"]`, `info["is_grasped"]`, etc. — use them in a custom
`gymnasium.Wrapper` or subclass `PandaLiftPixels` directly in your notebook.

## Layout

```
src/panda_lift_pixels/
├── contract.py    constants: OBS_SHAPE, ACTION_DIM, LIFT_HEIGHT, PARAM_LIMIT, REWARD_THRESHOLD…
├── env.py         PandaLiftPixels — the frozen gym.Env (observation + action + task)
├── export.py      export_model, selfcheck (SB3 → TorchScript)
└── grading.py     load_policy, check_contract, count_parameters, measure_latency, evaluate
```

## For the instructor

* **Calibrate before release.** `REWARD_THRESHOLD` in `contract.py` is a placeholder. Train a
  reference solution, then set the threshold to its median return minus a margin.
* **Hidden eval seeds.** `grading.evaluate` reads `EVAL_SEED_OFFSET` from the environment
  (default `0`). Set it as a GitHub Secret in the final grading workflow so students cannot
  overfit to the public seeds.
* **Versioning.** Bump `version` + tag (`vMAJOR.MINOR.PATCH`) for any change; student repos and
  the grader pin a tag, so a release is reproducible.
