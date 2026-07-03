"""The frozen contract for Project 2.

Every constant here is part of the graded contract. The package is installed from a pinned
git tag, so these values are identical on the student's Colab, in their CI, and in the
instructor's final grader. Students may *read* these (e.g. to know the threshold) but cannot
change what the grader uses.

Hidden at grading time: only the evaluation seeds (``EVAL_SEED_OFFSET`` env var, injected as a
GitHub Secret). Everything else here is public on purpose.
"""

# ---------------------------------------------------------------------------
# Observation contract — what the model receives as input
# ---------------------------------------------------------------------------
N_STACK = 4                                   # DQN-style frame stack
FRAME_HW = 96                                 # rendered frame is FRAME_HW x FRAME_HW
CHANNELS_PER_FRAME = 3                        # RGB
OBS_SHAPE = (N_STACK * CHANNELS_PER_FRAME, FRAME_HW, FRAME_HW)  # (12, 96, 96), channels-first
OBS_LOW = 0.0
OBS_HIGH = 1.0                                # observations are float32 in [0, 1] (already /255)

# ---------------------------------------------------------------------------
# Action contract — what the model must output
# ---------------------------------------------------------------------------
ACTION_DIM = 8                               # 7 joint position deltas + 1 gripper (joints control)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0                            # the grader clips outputs into [ACTION_LOW, ACTION_HIGH]

# ---------------------------------------------------------------------------
# Task definition — the behavior we grade (Lift: grasp the cube and hold it up)
# ---------------------------------------------------------------------------
BASE_ENV_ID = "PandaPickAndPlaceJoints-v3"
LIFT_HEIGHT = 0.10           # object-center height (m) above which the cube counts as "lifted"
GRASP_LIFT_OFF = 0.045       # object clearly off the table (resting center ≈ 0.02 m)
MAX_EPISODE_STEPS = 50       # grading horizon; canonical sparse reward integrates over this

# Canonical reward = 0.0 if (object lifted above LIFT_HEIGHT AND grasped) else -1.0, per step.
# Over MAX_EPISODE_STEPS this gives a return in [-MAX_EPISODE_STEPS, 0].

# ---------------------------------------------------------------------------
# Grading thresholds
# ---------------------------------------------------------------------------
PARAM_LIMIT = 10_000_000     # max number of parameters in the submitted model.pt
REWARD_THRESHOLD = -40.0     # PLACEHOLDER median cumulative reward to pass — set after calibration
LATENCY_BUDGET_S = 0.05      # max seconds per forward pass on CPU (keeps eval within time budget)

EVAL_EPISODES_CI = 30        # episodes run in student CI / local testing
EVAL_EPISODES_FINAL = 100    # episodes run in the instructor's final grading
