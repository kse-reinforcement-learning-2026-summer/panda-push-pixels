"""The frozen canonical environment for Project 2: ``PandaPushPixels``.

Task: push the cube onto the target marker — from pixels, with joints control. The gripper is
locked closed (this is panda-gym's stock Push task); the robot pushes with its closed fingers.

DO NOT MODIFY THIS FILE. The grader installs this package from a pinned git tag and uses its
own copy regardless of what is in the student's repository. Shape your training by reading the
privileged ``info`` dict (object/target positions, ee position, gripper-object contact, ...)
returned by ``reset()``/``step()`` in a thin ``gymnasium.Wrapper`` — the canonical reward is
sparse on purpose, so a naive ``model.learn()`` will not solve this without your own shaping.
Never touch the observation/action machinery below, or your model will receive a different
input distribution at grading time and fail to transfer.

Implementation note: this is a standalone ``gymnasium.Env`` that holds the panda-gym env as a
*private* attribute (composition), NOT a ``gymnasium.Wrapper`` chain. That is deliberate — a
wrapper chain leaks the base env's ``compute_reward`` through ``.unwrapped``, which makes
Stable-Baselines3's ``check_env`` misclassify this as a goal-conditioned env and crash.
"""

import collections

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  (registers Panda* environments with gymnasium)
from gymnasium import spaces

from .contract import (
    ACTION_DIM,
    BASE_ENV_ID,
    DISTANCE_THRESHOLD,
    DWELL_STEPS,
    MAX_EPISODE_STEPS,
    N_STACK,
    OBJECT_SIZE,
    OBS_SHAPE,
    STEP_PENALTY,
    SUCCESS_BONUS,
)

# Link indices of the two gripper fingers on the Panda body (verified against panda-gym 3.0.7).
# The gripper is locked closed for Push, so these act together as a single flat "paddle".
_LEFT_FINGER_LINK = 9
_RIGHT_FINGER_LINK = 10

# The stock target marker is translucent green — the same color as the cube, and TinyRenderer
# ignores alpha, so it would render as an indistinguishable solid green box. Repaint it red.
_TARGET_COLOR = np.array([0.9, 0.1, 0.1, 1.0])

# Camera: high 3/4 view over the table — both the cube and the target stay visible and
# color-distinct across the full spawn range; the arm sits off to the side, not occluding them.
_RENDER_DEFAULTS = dict(
    render_width=112,
    render_height=112,
    render_distance=0.85,
    render_target_position=[-0.15, 0.0, 0.15],
    render_yaw=50,
    render_pitch=-25,
)


class PandaPushPixels(gym.Env):
    """Pixel-observation Push task with joints control (frozen contract).

    Observation: ``Box(0, 1, (12, 112, 112), float32)`` — 4 stacked RGB frames, channels-first.
    Action:      ``Box(-1, 1, (7,), float32)`` — 7 joint position deltas (gripper locked closed).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_episode_steps=MAX_EPISODE_STEPS, render_kwargs=None):
        super().__init__()
        render_cfg = dict(_RENDER_DEFAULTS)
        if render_kwargs:
            render_cfg.update(render_kwargs)
        self._env = gym.make(BASE_ENV_ID, render_mode="rgb_array", **render_cfg)
        self._panda = self._env.unwrapped
        self._sim = self._panda.sim
        self._pc = self._sim.physics_client
        self._panda_id = self._sim._bodies_idx["panda"]
        self._object_id = self._sim._bodies_idx["object"]
        self._target_id = self._sim._bodies_idx["target"]
        self._pc.changeVisualShape(self._target_id, -1, rgbaColor=_TARGET_COLOR)

        self.observation_space = spaces.Box(0.0, 1.0, OBS_SHAPE, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

        self._frames = collections.deque(maxlen=N_STACK)
        self._t = 0                          # counts steps (one physics step per decision)
        self._max_steps = int(max_episode_steps)
        self._dwell = 0                      # consecutive steps the cube has stayed near the target

    # ------------------------------------------------------------------ #
    # FROZEN observation pipeline (identical in training and grading)
    # ------------------------------------------------------------------ #
    def _render_frame(self):
        frame = self._env.render()                    # (112, 112, 3) uint8 RGB
        # Keep RGB, channels-first: color is what tells the cube (green) apart from the target (red).
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)  # (3, 112, 112) uint8

    def _stacked_obs(self):
        stacked = np.concatenate(list(self._frames), axis=0)   # (12, 112, 112) uint8
        return (stacked.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------ #
    # Privileged state — for reward shaping by students; NOT the observation
    # ------------------------------------------------------------------ #
    def _is_touching(self):
        """Whether the (closed) gripper is in contact with the cube, from either finger link."""
        left = bool(self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_LEFT_FINGER_LINK
        ))
        right = bool(self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_RIGHT_FINGER_LINK
        ))
        return left or right

    def _privileged(self):
        obj = np.asarray(self._sim.get_base_position("object"), dtype=np.float32)
        tgt = np.asarray(self._sim.get_base_position("target"), dtype=np.float32)
        ee = np.asarray(self._panda.robot.get_ee_position(), dtype=np.float32)
        return {
            "object_position": obj,
            "target_position": tgt,
            "object_size": OBJECT_SIZE,
            "ee_position": ee,
            "object_to_target": float(np.linalg.norm(obj - tgt)),
            "ee_to_object": float(np.linalg.norm(ee - obj)),
            "is_touching": self._is_touching(),
        }

    # ------------------------------------------------------------------ #
    # Graded task — defines the behavior we want
    # ------------------------------------------------------------------ #
    def _step_outcome(self, p):
        """Advance the dwell counter and compute (success, reward, terminated, truncated).

        Success requires the cube to stay within DISTANCE_THRESHOLD of the target for DWELL_STEPS
        *consecutive* steps — a single-step graze from a fast-moving cube does not count (it would
        otherwise be trivial to "win" by smashing the cube through the target zone without ever
        placing it there). The one exception is the time limit itself: if the cube happens to be
        within the threshold right when the episode times out, that still counts — it would have
        dwelled long enough given a few more steps, and it would be unfair to punish a near-miss
        that was only cut off by the horizon.
        """
        close = p["object_to_target"] < DISTANCE_THRESHOLD
        self._dwell = self._dwell + 1 if close else 0
        timed_out = self._t >= self._max_steps

        dwell_success = self._dwell >= DWELL_STEPS
        success = dwell_success or (timed_out and close)
        reward = STEP_PENALTY + (SUCCESS_BONUS if success else 0.0)
        terminated = dwell_success
        truncated = (not terminated) and timed_out
        return success, reward, terminated, truncated

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)              # seeds self.np_random
        self._env.reset(seed=seed)
        self._t = 0
        self._dwell = 0

        frame = self._render_frame()
        self._frames.clear()
        for _ in range(N_STACK):              # padding_type="reset": repeat first frame
            self._frames.append(frame)

        obs = self._stacked_obs()
        info = {"is_success": False, **self._privileged()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        self._env.step(action)                     # one physics step per decision, no frame skip
        self._frames.append(self._render_frame())
        self._t += 1

        obs = self._stacked_obs()
        p = self._privileged()
        success, reward, terminated, truncated = self._step_outcome(p)
        info = {"is_success": success, **p}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
