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

* **Observation** — `Box(0, 255, (12, 112, 112), uint8)`: 4 stacked **RGB** frames (112×112),
  channels-first, raw `uint8`. Train with the SB3 default `normalize_images=True` (the standard
  image-RL setup) — SB3 divides by 255 inside the policy, so the CNN sees `[0, 1]`. **uint8 keeps
  the rollout/replay buffer 4× lighter than float32** (matters for large PPO rollouts and SAC).
  **Why RGB:** the cube is green and the target marker is red — color is what tells them apart;
  in the stock panda-gym scene the target is a translucent *green* box (same hue as the cube),
  so we repaint it red for this project.
* **Action** — `Box(-1, 1, (7,), float32)`: 7 joint position deltas.
* **Canonical reward** — `STEP_PENALTY` (`-1.0`) every step, plus `SUCCESS_BONUS` (`+50.0`) added
  on the step success is achieved. Success requires the cube to stay within `DISTANCE_THRESHOLD`
  (5 cm) of the target for `DWELL_STEPS` (5) *consecutive* steps — a single-step graze from a
  fast-moving cube doesn't count, so "fling the cube through the target zone" isn't a shortcut.
  The one exception: if the cube is within the threshold right when the 50-step time limit hits,
  that still counts (it would have dwelled long enough given a few more steps). Reward is a
  training signal — grading is tiered on two behavioral metrics instead (below).

### Grading — tiered rubric (5 / 10 / 15 points, cumulative)

Both tier metrics come from the same deterministic evaluation episodes
(`grading.evaluate`/`evaluate_policy`), and both live in `contract.py`:

* **5 pts** — the submitted notebook imports one of `ALLOWED_SB3_ALGOS` (`A2C`, `PPO`, `DDPG`,
  `TD3`, `SAC`) and calls `.learn(...)` in the student's own sections (checked by
  `grading.notebook_trains_sb3`, pure JSON parsing — the notebook is never executed), **and**
  `model.pt` loads, satisfies the I/O contract, and has `<= PARAM_LIMIT` parameters.
* **10 pts** — `touch_rate >= TOUCH_RATE_THRESHOLD` (`0.80`): the cube's centre moves more than
  `TOUCH_DISPLACEMENT` (`0.01` m) from its spawn position at some point in the episode — the agent
  found and nudged the cube, direction irrelevant.
* **15 pts** — `success_rate >= SUCCESS_RATE_THRESHOLD` (`0.50`): the push actually succeeds
  (reaches the target and dwells, per the canonical reward above).

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
pip install "panda-push-pixels @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v11.0.0"

# Training (Colab/Kaggle) — keep the platform's GPU torch, add the SB3 stack:
pip install "panda-push-pixels[train] @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v11.0.0"
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
pip install "panda-push-pixels[train] @ git+https://github.com/kse-reinforcement-learning-2026-summer/panda-push-pixels.git@v11.0.0"

# 4. Verify
python -c "import gymnasium as gym, panda_push_pixels; \
env = gym.make('PandaPushPixels-v0'); obs, info = env.reset(seed=0); \
print('OK', obs.shape, obs.dtype); env.close()"
# Expected: OK (12, 112, 112) uint8
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
    contract,            # all contract constants (shapes, limits, tiered thresholds)
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
├── contract.py    constants: OBS_SHAPE, ACTION_DIM, DISTANCE_THRESHOLD, PARAM_LIMIT, tiered thresholds…
├── env.py         PandaPushPixels — the frozen gym.Env (observation + action + task)
├── viz.py         render_episode, save_video — rollout + visualization helpers (not part of grading)
├── export.py      export_model, extract_actor, selfcheck (SB3 → TorchScript)
└── grading.py     load_policy, check_contract, count_parameters, measure_latency, evaluate,
                   evaluate_policy (returns touch_rate + success_rate), notebook_trains_sb3
```

## For the instructor

* **Tiered thresholds are fixed by rubric, not calibration.** `TOUCH_RATE_THRESHOLD` (`0.80`) and
  `SUCCESS_RATE_THRESHOLD` (`0.50`) in `contract.py` are the course's stated 10-pt/15-pt bars, not
  placeholders to be loosened to match a reference run. A reference solution trained end-to-end
  reached ~100% touch / ~13% push success — 15 pts is intended to be very hard to reach.
* **Hidden eval seeds.** `grading.evaluate` reads `EVAL_SEED_OFFSET` from the environment
  (default `0`). Set it as a GitHub Secret in the final grading workflow so students cannot
  overfit to the public seeds.
* **Versioning.** Bump `version` + tag (`vMAJOR.MINOR.PATCH`) for any change; student repos and
  the grader pin a tag, so a release is reproducible.
