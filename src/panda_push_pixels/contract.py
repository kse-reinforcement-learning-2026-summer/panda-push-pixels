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
FRAME_HW = 112                                # rendered frame is FRAME_HW x FRAME_HW
CHANNELS_PER_FRAME = 3                        # RGB (the cube is green, the target is red — color matters)
OBS_SHAPE = (N_STACK * CHANNELS_PER_FRAME, FRAME_HW, FRAME_HW)  # (12, 112, 112), channels-first
OBS_LOW = 0
OBS_HIGH = 255                               # observations are uint8 in [0, 255] (raw RGB, channels-first);
                                             # SB3's default normalize_images=True does the /255 inside the policy.
                                             # uint8 keeps the rollout/replay buffer 4x lighter than float32.

# ---------------------------------------------------------------------------
# Action contract — what the model must output
# ---------------------------------------------------------------------------
ACTION_DIM = 7                               # 7 joint position deltas (joints control; gripper stays closed)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0                            # the grader clips outputs into [ACTION_LOW, ACTION_HIGH]

# ---------------------------------------------------------------------------
# Task definition — the behavior we grade (Push: push the cube onto the target)
# ---------------------------------------------------------------------------
BASE_ENV_ID = "PandaPushJoints-v3"
OBJECT_SIZE = 0.04           # cube (and target marker) side length (m)
DISTANCE_THRESHOLD = 0.05    # success: object-to-target center distance (m) below this
MAX_EPISODE_STEPS = 50       # grading horizon (one physics step per decision, no frame skip)
DWELL_STEPS = 5              # consecutive close steps required for success (rejects a fly-through graze)
STEP_PENALTY = -1.0          # reward on every step
SUCCESS_BONUS = 50.0         # ADDED on top of STEP_PENALTY on the step success is achieved

# Every step: reward = STEP_PENALTY, plus SUCCESS_BONUS on the step success is achieved (so a
# successful step's reward is STEP_PENALTY + SUCCESS_BONUS). Success requires the cube to stay
# within DISTANCE_THRESHOLD of the target for DWELL_STEPS consecutive steps — a single-step touch
# from a fast-moving cube does not count. The one exception: if the cube is within the threshold
# right when the episode times out, that still counts (it would have dwelled long enough given a
# few more steps). The episode always runs to MAX_EPISODE_STEPS unless the dwell condition fires
# first. A total failure returns exactly STEP_PENALTY * MAX_EPISODE_STEPS; the fastest possible
# success (dwelling from step 1) returns STEP_PENALTY * DWELL_STEPS + SUCCESS_BONUS.

# ---------------------------------------------------------------------------
# Grading thresholds — tiered rubric (5 / 10 / 15 points)
# ---------------------------------------------------------------------------
PARAM_LIMIT = 4_000_000      # max number of parameters in the submitted model.pt (5-pt gate)

# Tier metrics, both measured over the same deterministic evaluation episodes:
TOUCH_DISPLACEMENT = 0.01    # the cube counts as "moved" if its centre leaves the spawn by > 1 cm
TOUCH_RATE_THRESHOLD = 0.80  # 10-pt tier: fraction of episodes in which the agent MOVED the cube
SUCCESS_RATE_THRESHOLD = 0.5 # 15-pt tier: fraction of episodes that SUCCEED (push onto target + dwell)

# The SB3 algorithms taught in the course — the submitted notebook must import one of these and call
# .learn() (5-pt gate). The naive PPO baseline shipped in the template's prewritten sections (1-2)
# does NOT count: the check only scans code from the first student section (## 3 onward).
ALLOWED_SB3_ALGOS = ("A2C", "PPO", "DDPG", "TD3", "SAC")

LATENCY_BUDGET_S = 0.05      # max seconds per forward pass on CPU (keeps eval within time budget)

EVAL_EPISODES_CI = 30        # episodes run in student CI / local testing
EVAL_EPISODES_FINAL = 100    # episodes run in the instructor's final grading
