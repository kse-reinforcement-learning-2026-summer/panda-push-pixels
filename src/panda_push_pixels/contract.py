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
OBS_LOW = 0.0
OBS_HIGH = 1.0                                # observations are float32 in [0, 1] (already /255)

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
# Grading thresholds
# ---------------------------------------------------------------------------
PARAM_LIMIT = 10_000_000     # max number of parameters in the submitted model.pt
SUCCESS_RATE_THRESHOLD = 0.5 # PLACEHOLDER fraction of eval episodes that must succeed — set after calibration
LATENCY_BUDGET_S = 0.05      # max seconds per forward pass on CPU (keeps eval within time budget)

EVAL_EPISODES_CI = 30        # episodes run in student CI / local testing
EVAL_EPISODES_FINAL = 100    # episodes run in the instructor's final grading
