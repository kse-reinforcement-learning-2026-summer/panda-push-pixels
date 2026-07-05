# panda-push-pixels

Frozen pixel-observation **Panda Push** environment and autograder for **KSE Reinforcement
Learning — Coding Project 2**.

This package is the single source of truth for the project contract. It is installed from a
pinned git tag, so the environment and grader are byte-identical on the student's Colab, in
their GitHub Actions CI, and in the instructor's final grading run.

## The task

Push the cube onto the target marker — observed **only from pixels** (4 stacked RGB frames,
DQN-style), controlled at the **joint** level (7 joint position deltas; the gripper stays
closed and the policy solves its own IK).

* **Observation** — `Box(0, 1, (12, 112, 112), float32)`: 4 stacked **RGB** frames (112×112),
  channels-first, already normalized to `[0, 1]`. (Do **not** normalize again in your model.)
  **Why RGB:** the cube is green and the target marker is red — color is what tells them apart;
  in the stock panda-gym scene the target is a translucent *green* box (same hue as the cube),
  so we repaint it red for this project.
* **Action** — `Box(-1, 1, (7,), float32)`: 7 joint position deltas.
* **Canonical reward** — sparse: `0.0` on a step where the cube is within `DISTANCE_THRESHOLD`
  (5 cm) of the target, else `-1.0`. The episode **terminates the instant the cube reaches the
  target**; otherwise it truncates at 50 steps. The graded metric is the **median cumulative
  reward** (i.e. `-`steps-to-solve, or `-50` on a timeout).

### Why this is harder than it looks

A naive `model.learn()` on this environment will not work — that is by design. Solving it
requires four separate skills:

1. **Perception.** The cube's and the target's 3D positions must be read off a 112×112 RGB
   image — there is no privileged state in the observation itself.
2. **Inverse kinematics.** The action is 7 joint deltas, not an end-effector position — the
   policy has to learn how joint motion maps to end-effector motion on its own.
3. **Reward design / curriculum.** The canonical reward is sparse (`0`/`-1`), so early in
   training almost every episode returns the same `-50` regardless of behavior — there is no
   gradient towards "get closer to the cube" unless you build one yourself. You will need your
   own shaped reward and/or curriculum, using the privileged `info` dict (see below), to make
   the task learnable in a reasonable number of steps.
4. **Contact physics.** Pushing only works through correct contact — friction, contact force,
   and how the cube slides once nudged all have to be exploited, not just avoided.

## Install

```bash
# Grading / evaluation only (CI, local tests) — no Stable-Baselines3:
pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "panda-push-pixels @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v7.0.0"

# Training (Colab/Kaggle) — keep the platform's GPU torch, add the SB3 stack:
pip install "panda-push-pixels[train] @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v7.0.0"
```

Requires **Python 3.11+** (panda-gym pins `numpy<2`; pybullet builds from source on 3.13+).
On Colab (3.12) pybullet builds from source the first time (~3–6 min); install `numpy<2` first.

### Local install on macOS (Apple Silicon & Intel)

On a Mac, installing `pybullet` from PyPI tries to **compile from source** (needs Xcode Command
Line Tools and often fails on Python 3.13). The reliable route is to grab a **prebuilt
`pybullet`** from `conda-forge`, then `pip install` the rest. Tested on Apple Silicon (arm64),
Python 3.12.

```bash
# 1. Create and activate an isolated environment (Python 3.11 or 3.12 — NOT 3.13)
conda create -n rl-project2 python=3.12 -y
conda activate rl-project2

# 2. Prebuilt pybullet + numpy<2 from conda-forge (avoids the source build)
conda install -c conda-forge "pybullet=3.25" "numpy<2" -y

# 3. Install the project (training stack: Stable-Baselines3, etc.)
pip install "panda-push-pixels[train] @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v7.0.0"

# 4. Verify
python -c "import gymnasium as gym, panda_push_pixels; \
env = gym.make('PandaPushPixels-v0'); obs, info = env.reset(seed=0); \
print('OK', obs.shape, obs.dtype); env.close()"
# Expected: OK (12, 112, 112) float32
```

Notes:
* Don't have conda? Install [Miniforge](https://github.com/conda-forge/miniforge) (native arm64).
* Training on Mac runs on **CPU** (SB3 uses `device="cpu"`/`"auto"`; MPS is not used). It works
  but is slower than Colab's GPU — use a small `total_timesteps` when iterating locally.
* `pip install` (step 3) will **not** rebuild `pybullet` because the conda-forge binary already
  satisfies `panda-gym`'s requirement. If pip ever tries to, re-run step 2 first.


## Public API

```python
import gymnasium as gym
import panda_push_pixels
from panda_push_pixels import (
    make_eval_env,      # the exact frozen env the grader uses
    render_episode,      # roll out one episode (random or your policy) -> frames + trajectory
    save_video,          # save render_episode(...)["frames"] to a .mp4/.gif
    export_model,        # SB3 model  ->  standalone TorchScript model.pt (actor only)
    extract_actor,       # SB3 model  ->  actor nn.Module (A2C/PPO/DDPG/TD3/SAC)
    selfcheck,           # assert the exported model.pt == sb3_model.predict(deterministic=True)
    grading,             # grading.evaluate / evaluate_policy / check_contract / count_parameters / measure_latency
    contract,            # all contract constants (shapes, limits, threshold)
)

# Use either gym.make or make_eval_env
env = gym.make("PandaPushPixels-v0")
# env = make_eval_env()  # equivalent

render_episode(env)  # sanity-check: one rollout with random actions
```

**For training:** wrap the env with your own reward shaping. The frozen env exposes
`info["object_position"]`, `info["target_position"]`, `info["object_size"]`,
`info["ee_position"]`, `info["object_to_target"]`, `info["ee_to_object"]`, `info["is_touching"]`
— use them in a custom `gymnasium.Wrapper` or subclass `PandaPushPixels` directly in your
notebook.

## Layout

```
src/panda_push_pixels/
├── contract.py    constants: OBS_SHAPE, ACTION_DIM, DISTANCE_THRESHOLD, PARAM_LIMIT, REWARD_THRESHOLD…
├── env.py         PandaPushPixels — the frozen gym.Env (observation + action + task)
├── viz.py         render_episode, save_video — rollout + visualization helpers (not part of grading)
├── export.py      export_model, extract_actor, selfcheck (SB3 → TorchScript)
└── grading.py     load_policy, check_contract, count_parameters, measure_latency, evaluate, evaluate_policy
```

## For the instructor

* **Calibrate before release.** `REWARD_THRESHOLD` in `contract.py` is a placeholder. Train a
  reference solution, then set the threshold to its median return minus a margin.
* **Hidden eval seeds.** `grading.evaluate` reads `EVAL_SEED_OFFSET` from the environment
  (default `0`). Set it as a GitHub Secret in the final grading workflow so students cannot
  overfit to the public seeds.
* **Versioning.** Bump `version` + tag (`vMAJOR.MINOR.PATCH`) for any change; student repos and
  the grader pin a tag, so a release is reproducible.
