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
# Per-finger half-widths (m): closed onto the ~0.04 m cube (grasp) vs fully open (reach).
_GRASP_FINGER_HALF = 0.019
_OPEN_FINGER_HALF = 0.04
# Default ee-start box (m) for reach augmentation / in-air grasp: over the cube-spawn square, in the air.
_DEFAULT_EE_BOX = {"x": (-0.13, 0.13), "y": (-0.13, 0.13), "z": (0.08, 0.22)}
_CUBE_SIZE = 0.04            # cube side (m); an OPEN gripper must stay >= this so it can't push the cube
_MAX_OPEN = 0.08             # maximum fingers_width

# Camera: frontal view framed on the workspace — the gripper sits centred above the cube, so the
# gripper<->cube alignment is read head-on (clearer for grasping); table fills the frame, floor trimmed.
_RENDER_DEFAULTS = dict(
    render_width=112,
    render_height=112,
    render_distance=1.1,
    render_target_position=[-0.35, 0.0, 0.07],
    render_yaw=90,
    render_pitch=-25,
)


class PandaLiftPixels(gym.Env):
    """Pixel-observation Lift task with joints control (frozen contract).

    Observation: ``Box(0, 1, (12, 112, 112), float32)`` — 4 stacked RGB frames, channels-first.
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
        frame = self._env.render()                    # (112, 112, 3) uint8 RGB
        # Keep RGB, channels-first: the cube is green and pops in colour (muddy in grayscale).
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)  # (3, 112, 112) uint8

    def _stacked_obs(self):
        stacked = np.concatenate(list(self._frames), axis=0)   # (12, 112, 112) uint8
        return (stacked.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------ #
    # Privileged state — for reward shaping by students; NOT the observation
    # ------------------------------------------------------------------ #
    def _contacts(self, object_height):
        """Return (left_touch, right_touch, is_touching, is_grasped).

        left_touch / right_touch: whether that single finger is in contact with the cube —
                     a fine-grained shaping signal (1 finger < 2 fingers gives partial credit).
        is_touching: BOTH fingers are in contact with the cube (pure contact, no height gate) —
                     an early, informative shaping signal available before the cube is lifted.
        is_grasped:  is_touching AND the cube is already off the table (object_height > 0.045) —
                     this is the stricter signal used by the success criterion.
        """
        left = bool(self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_LEFT_FINGER_LINK
        ))
        right = bool(self._pc.getContactPoints(
            bodyA=self._panda_id, bodyB=self._object_id, linkIndexA=_RIGHT_FINGER_LINK
        ))
        touching = left and right
        return left, right, touching, (touching and object_height > GRASP_LIFT_OFF)

    def _privileged(self):
        obj = np.asarray(self._sim.get_base_position("object"), dtype=np.float32)
        ee = np.asarray(self._panda.robot.get_ee_position(), dtype=np.float32)
        vel = np.asarray(self._sim.get_base_velocity("object"), dtype=np.float32)
        left_touch, right_touch, touching, grasped = self._contacts(float(obj[2]))
        return {
            "object_position": obj,
            "ee_position": ee,
            "object_velocity": vel,
            "object_height": float(obj[2]),
            "gripper_to_object": float(np.linalg.norm(ee - obj)),
            "fingers_width": float(self._panda.robot.get_fingers_width()),
            "left_finger_touch": left_touch,
            "right_finger_touch": right_touch,
            "n_fingers_touching": int(left_touch) + int(right_touch),
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

        # Optional start-state overrides (training curriculum; grading passes none => standard start).
        # Precedence: start_lifted > start_grasped > ee_start.
        if options.get("start_lifted", False):
            # Grasp IN THE AIR: sample an ee position in a box, IK the closed gripper there, and
            # place the cube in the gripper. Bootstraps lift/hold from varied poses/heights.
            x, y, z = self._sample_ee(options.get("ee_start", _DEFAULT_EE_BOX))
            ee = self._ik_set_arm(x, y, z, _GRASP_FINGER_HALF)
            self._sim.set_base_pose("object", ee, np.array([0.0, 0.0, 0.0, 1.0]))
        elif options.get("start_grasped", False):
            # Grasp the cube where it sits ON THE TABLE (default h=0.02) or at a given/random height:
            # IK the arm down onto the cube's xy, close the fingers around it. Bootstraps the LIFT.
            obj = self._sim.get_base_position("object")
            height = self._sample_scalar(options.get("grasp_height", 0.02))
            ee = self._ik_set_arm(float(obj[0]), float(obj[1]), height, _GRASP_FINGER_HALF)
            self._sim.set_base_pose("object", ee, np.array([0.0, 0.0, 0.0, 1.0]))
        elif options.get("start_reached", False):
            # Gripper positioned AT the cube but OPEN (>= cube width, never narrower -> can't push the
            # cube): the approach is done, the agent only has to CLOSE. The cube sits on the table
            # under the (open) gripper -- placed at the achieved ee xy so IK error can't leave a gap.
            obj = self._sim.get_base_position("object")
            height = self._sample_scalar(options.get("grasp_height", 0.02))
            half = self._sample_finger_half(options.get("gripper_width", (_CUBE_SIZE, _MAX_OPEN)))
            ee = self._ik_set_arm(float(obj[0]), float(obj[1]), height, half)
            self._set_object_xy(float(ee[0]), float(ee[1]))
        elif "ee_start" in options:
            # Non-grasped: place the gripper (open, width from gripper_width) at a given/random ee
            # position (reach-stage variety); the cube stays on the table.
            x, y, z = self._sample_ee(options["ee_start"])
            half = self._sample_finger_half(options.get("gripper_width", _MAX_OPEN))
            self._ik_set_arm(x, y, z, half)

    def _set_object_xy(self, x, y):
        self._sim.set_base_pose(
            "object", np.array([x, y, 0.02]), np.array([0.0, 0.0, 0.0, 1.0])
        )

    def _sample_scalar(self, spec):
        """A float, or a ``(lo, hi)`` pair sampled uniformly with the env's seeded RNG."""
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return float(self.np_random.uniform(float(spec[0]), float(spec[1])))
        return float(spec)

    def _sample_finger_half(self, width_spec):
        """Total gripper width (float or ``(lo, hi)``) -> per-finger half-width."""
        return self._sample_scalar(width_spec) / 2.0

    def _sample_ee(self, spec):
        """ee target: ``[x, y, z]`` fixed, or ``{"x":(lo,hi), "y":(lo,hi), "z":(lo,hi)}`` sampled."""
        if isinstance(spec, dict):
            return (self._sample_scalar(spec.get("x", 0.0)),
                    self._sample_scalar(spec.get("y", 0.0)),
                    self._sample_scalar(spec.get("z", 0.15)))
        return (float(spec[0]), float(spec[1]), float(spec[2]))

    def _ik_set_arm(self, x, y, z, finger_half):
        """IK the arm so the (gripper-down) ee reaches (x, y, z); set the finger half-widths.

        The arm is at its neutral (gripper-down) pose when this runs, so we reuse that orientation
        as the IK target. Returns the achieved ee position (IK is approximate, ~2 cm)."""
        robot = self._panda.robot
        down_orn = np.array(self._pc.getLinkState(self._panda_id, robot.ee_link)[1])
        q = robot.inverse_kinematics(
            link=robot.ee_link, position=np.array([x, y, z], dtype=np.float64), orientation=down_orn
        )
        angles = np.asarray(q[: len(robot.joint_indices)], dtype=np.float64)
        angles[-2:] = finger_half                           # both finger joints (open or closed)
        robot.set_joint_angles(angles)
        return np.asarray(robot.get_ee_position(), dtype=np.float64)

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
