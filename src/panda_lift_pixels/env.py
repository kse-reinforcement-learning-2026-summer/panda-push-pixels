"""The frozen canonical environment for Project 2: ``PandaLiftPixels``.

Task: grasp the cube and lift it above the table, then keep it from falling — from pixels,
with joints control.

DO NOT MODIFY THIS FILE. The grader installs this package from a pinned git tag and uses its
own copy regardless of what is in the student's repository. Shape your training by:
  * reading the privileged ``info`` dict (object/gripper positions, is_touching, is_grasped, ...)
    in a thin ``gymnasium.Wrapper`` to compute any reward you like, and/or
  * passing curriculum ``options`` to ``reset()`` (object placement, lift bar, start-grasped).
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
    GRASP_LIFT_OFF,
    LIFT_HEIGHT,
    MAX_EPISODE_STEPS,
    N_STACK,
    OBS_SHAPE,
)

# Link indices of the two gripper fingers on the Panda body (verified against panda-gym 3.0.7).
_LEFT_FINGER_LINK = 9
_RIGHT_FINGER_LINK = 10

# Camera: 3/4 view framed on the workspace — shows the whole arm (all joints/links, needed for
# joints control), the gripper, table, and cube; background/floor trimmed out.
_RENDER_DEFAULTS = dict(
    render_width=96,
    render_height=96,
    render_distance=1.0,
    render_target_position=[-0.15, 0.0, 0.15],
    render_yaw=50,
    render_pitch=-20,
)


class PandaLiftPixels(gym.Env):
    """Pixel-observation Lift task with joints control (frozen contract).

    Observation: ``Box(0, 1, (12, 96, 96), float32)`` — 4 stacked RGB frames, channels-first.
    Action:      ``Box(-1, 1, (8,), float32)`` — 7 joint position deltas + gripper.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_episode_steps=MAX_EPISODE_STEPS, lift_height=LIFT_HEIGHT,
                 action_repeat=2, render_kwargs=None):
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

        self.observation_space = spaces.Box(0.0, 1.0, OBS_SHAPE, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

        self._frames = collections.deque(maxlen=N_STACK)
        self._t = 0                          # counts agent *decisions*, not physics steps
        self._max_steps = int(max_episode_steps)
        self._action_repeat = max(1, int(action_repeat))
        self._default_lift_height = float(lift_height)
        self._lift_height = self._default_lift_height

    # ------------------------------------------------------------------ #
    # FROZEN observation pipeline (identical in training and grading)
    # ------------------------------------------------------------------ #
    def _render_frame(self):
        frame = self._env.render()                    # (96, 96, 3) uint8 RGB
        # Keep RGB, channels-first: the cube is green and pops in colour (muddy in grayscale).
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)  # (3, 96, 96) uint8

    def _stacked_obs(self):
        stacked = np.concatenate(list(self._frames), axis=0)   # (12, 96, 96) uint8
        return (stacked.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------ #
    # Privileged state — for reward shaping by students; NOT the observation
    # ------------------------------------------------------------------ #
    def _contacts(self, object_height):
        """Return (is_touching, is_grasped).

        is_touching: BOTH fingers are in contact with the cube (pure contact, no height gate) —
                     an early, informative shaping signal available before the cube is lifted.
        is_grasped:  is_touching AND the cube is already off the table (object_height > 0.045) —
                     this is the stricter signal used by the success criterion.
        """
        left = self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_LEFT_FINGER_LINK
        )
        right = self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_RIGHT_FINGER_LINK
        )
        touching = bool(left) and bool(right)
        return touching, (touching and object_height > GRASP_LIFT_OFF)

    def _privileged(self):
        obj = np.asarray(self._sim.get_base_position("object"), dtype=np.float32)
        ee = np.asarray(self._panda.robot.get_ee_position(), dtype=np.float32)
        vel = np.asarray(self._sim.get_base_velocity("object"), dtype=np.float32)
        touching, grasped = self._contacts(float(obj[2]))
        return {
            "object_position": obj,
            "ee_position": ee,
            "object_velocity": vel,
            "object_height": float(obj[2]),
            "gripper_to_object": float(np.linalg.norm(ee - obj)),
            "fingers_width": float(self._panda.robot.get_fingers_width()),
            "is_touching": touching,
            "is_grasped": grasped,
        }

    # ------------------------------------------------------------------ #
    # Graded task — defines the behavior we want
    # ------------------------------------------------------------------ #
    def _is_success(self, p):
        return bool(p["object_height"] > self._lift_height and p["is_grasped"])

    def _canonical_reward(self, p):
        return 0.0 if self._is_success(p) else -1.0

    def _hide_target(self):
        # Lift has no place-target; move the goal marker out of camera view so it is not a
        # distractor in the observation (TinyRenderer ignores alpha, so we cannot fade it out).
        self._sim.set_base_pose(
            "target", np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0])
        )

    # ------------------------------------------------------------------ #
    # Curriculum hook — options is None at grading => default distribution
    # ------------------------------------------------------------------ #
    def _apply_curriculum(self, options):
        # Reset to the graded defaults first so a previous episode's overrides never leak.
        self._lift_height = self._default_lift_height
        if not options:
            return
        self._lift_height = float(options.get("lift_height", self._default_lift_height))

        if "object_position" in options:
            xy = options["object_position"]
            self._set_object_xy(float(xy[0]), float(xy[1]))
        elif "object_xy_range" in options:
            r = float(options["object_xy_range"])
            xy = self.np_random.uniform(-r, r, size=2)
            self._set_object_xy(float(xy[0]), float(xy[1]))

        if options.get("start_grasped", False):
            # Teleport the cube to the gripper so the agent only has to close + lift (bootstrap).
            ee = np.asarray(self._panda.robot.get_ee_position(), dtype=np.float32)
            self._sim.set_base_pose(
                "object", ee.astype(np.float64), np.array([0.0, 0.0, 0.0, 1.0])
            )

    def _set_object_xy(self, x, y):
        self._sim.set_base_pose(
            "object", np.array([x, y, 0.02]), np.array([0.0, 0.0, 0.0, 1.0])
        )

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)              # seeds self.np_random
        self._env.reset(seed=seed)
        self._hide_target()
        self._apply_curriculum(options)
        self._t = 0

        frame = self._render_frame()
        self._frames.clear()
        for _ in range(N_STACK):              # padding_type="reset": repeat first frame
            self._frames.append(frame)

        obs = self._stacked_obs()
        info = {"is_success": False, **self._privileged()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        # Repeat action for _action_repeat physics steps; render only the final state.
        # This skips intermediate renders (the render bottleneck) while covering more
        # simulation time per agent decision.
        for _ in range(self._action_repeat):
            self._env.step(action)
        self._frames.append(self._render_frame())  # one render per decision
        self._t += 1                               # count decisions

        obs = self._stacked_obs()
        p = self._privileged()
        reward = self._canonical_reward(p)
        terminated = False                    # never end early: we reward HOLDING over the horizon
        truncated = self._t >= self._max_steps
        info = {"is_success": self._is_success(p), **p}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
