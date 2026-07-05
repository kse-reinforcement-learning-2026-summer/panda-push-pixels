"""Self-tests for the panda-lift-pixels package (run in the package repo's own CI).

These verify the frozen contract and the grading pipeline. They train a tiny model on CPU, so
they are slow-ish but self-contained. Requires the ``[train]`` extra (Stable-Baselines3).
"""

import os

import gymnasium as gym
import numpy as np
import pytest
import torch

import panda_lift_pixels
from panda_lift_pixels import contract, export_model, make_eval_env, selfcheck
from panda_lift_pixels import grading

MODEL_PATH = "/tmp/_plp_selftest_model.pt"


def test_env_observation_contract():
    env = make_eval_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == contract.OBS_SHAPE
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert tuple(env.action_space.shape) == (contract.ACTION_DIM,)
    assert "is_grasped" in info and "object_height" in info
    env.close()


def test_episode_horizon_and_reward():
    """Horizon is MAX_EPISODE_STEPS agent *decisions* (not physics steps)."""
    env = make_eval_env()
    env.reset(seed=0)
    rewards, done, steps = [], False, 0
    while not done:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        rewards.append(r)
        steps += 1
        done = term or trunc
    assert steps == contract.MAX_EPISODE_STEPS   # 50 decisions regardless of action_repeat
    assert set(np.unique(rewards)).issubset({-1.0, 0.0})
    env.close()


def test_action_repeat_default():
    """Default action_repeat=2: each env.step() covers 2 physics steps (1 render)."""
    from panda_lift_pixels import PandaLiftPixels
    env = PandaLiftPixels()
    assert env._action_repeat == contract.ACTION_REPEAT == 2
    obs1, _ = env.reset(seed=0)
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert obs1.shape == obs2.shape == contract.OBS_SHAPE
    env.close()


def test_curriculum_start_grasped_on_table():
    """start_grasped: cube gripped where it spawns ON THE TABLE (h<threshold => must still be lifted)."""
    env = make_eval_env()
    env.reset(seed=2, options={"start_grasped": True})
    _, _, _, _, info = env.step(np.zeros(contract.ACTION_DIM, dtype=np.float32))  # 1 step registers contact
    assert info["n_fingers_touching"] == 2
    assert info["object_height"] < contract.GRASP_LIFT_OFF      # on the table -> agent must lift it
    assert info["gripper_to_object"] < 0.03                     # cube held in the gripper
    env.close()


def test_curriculum_start_lifted_in_air():
    """start_lifted: cube gripped IN THE AIR (above the lift-off height), from a sampled ee box."""
    env = make_eval_env()
    env.reset(seed=3, options={"start_lifted": True})
    _, _, _, _, info = env.step(np.zeros(contract.ACTION_DIM, dtype=np.float32))
    assert info["n_fingers_touching"] == 2
    assert info["object_height"] > contract.GRASP_LIFT_OFF      # already lifted -> agent must hold
    env.close()


def test_curriculum_ee_start_reach():
    """ee_start (no grasp): gripper placed at a target with a chosen width; cube stays on the table."""
    env = make_eval_env()
    _, info = env.reset(seed=4, options={"ee_start": [0.10, -0.08, 0.18], "gripper_width": 0.05})
    assert abs(info["fingers_width"] - 0.05) < 0.01             # width honoured (randomisable via range)
    assert info["object_height"] < contract.GRASP_LIFT_OFF      # cube on the table, not grasped
    assert np.linalg.norm(info["ee_position"] - np.array([0.10, -0.08, 0.18])) < 0.06  # IK ~near target
    env.close()


def test_curriculum_start_reached():
    """start_reached: OPEN gripper (>= cube width) at the cube on the table; agent only has to close."""
    env = make_eval_env()
    _, info = env.reset(seed=6, options={"start_reached": True})
    assert info["object_height"] < contract.GRASP_LIFT_OFF      # cube on the table, not lifted
    assert info["gripper_to_object"] < 0.03                     # gripper is AT the cube
    assert info["fingers_width"] >= 0.039                       # open at least ~cube width (not grasping)
    env.close()


def test_is_touching_present_and_pure_contact():
    """info exposes a pure-contact is_touching signal (grasp without the height gate)."""
    env = make_eval_env()
    _, info = env.reset(seed=0)
    # both keys present, boolean-typed
    assert "is_touching" in info and "is_grasped" in info
    assert isinstance(bool(info["is_touching"]), bool)
    # per-finger contact ladder exposed for grasp shaping
    assert {"left_finger_touch", "right_finger_touch", "n_fingers_touching"} <= set(info)
    for _ in range(20):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        # is_grasped implies is_touching (grasped = touching AND lifted), never the reverse constraint
        assert not (info["is_grasped"] and not info["is_touching"])
        # n_fingers_touching in {0,1,2} and consistent with the per-finger / both-finger signals
        n = info["n_fingers_touching"]
        assert n == int(info["left_finger_touch"]) + int(info["right_finger_touch"])
        assert n in (0, 1, 2)
        assert info["is_touching"] == (n == 2)
        if term or trunc:
            break
    env.close()


@pytest.fixture(scope="module")
def tiny_model():
    from stable_baselines3 import SAC

    env = gym.make("PandaLiftPixels-v0", max_episode_steps=50)
    model = SAC(
        "CnnPolicy", env, buffer_size=400, learning_starts=40, train_freq=4, batch_size=32,
        policy_kwargs=dict(normalize_images=False), device="cpu", verbose=0,
    )
    model.learn(total_timesteps=120)
    export_model(model, MODEL_PATH)
    yield model
    env.close()
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)


def test_export_matches_sb3(tiny_model):
    assert selfcheck(tiny_model, MODEL_PATH, n_steps=5) < 1e-4


@pytest.mark.parametrize("algo_name", ["A2C", "PPO", "DDPG", "TD3", "SAC"])
def test_extract_actor_all_algorithms(algo_name):
    """extract_actor must reproduce model.predict(deterministic=True) for every allowed algorithm.

    Untrained (random) weights are enough — we only check the extracted actor path matches SB3.
    """
    import stable_baselines3 as sb3
    from panda_lift_pixels import extract_actor

    Algo = getattr(sb3, algo_name)
    env = gym.make("PandaLiftPixels-v0")
    kwargs = dict(policy_kwargs=dict(normalize_images=False), device="cpu", verbose=0)
    if algo_name in ("DDPG", "TD3", "SAC"):
        kwargs["buffer_size"] = 200  # tiny: avoid allocating a huge image replay buffer
    model = Algo("CnnPolicy", env, **kwargs)

    actor = extract_actor(model).eval()
    obs_batch = np.random.rand(4, *contract.OBS_SHAPE).astype(np.float32)
    with torch.no_grad():
        actions_actor = actor(torch.as_tensor(obs_batch)).numpy()
    actions_sb3 = np.array([model.predict(o, deterministic=True)[0] for o in obs_batch])

    assert np.abs(actions_actor - actions_sb3).max() < 1e-4
    env.close()


def test_evaluate_policy_matches_evaluate(tiny_model):
    """evaluate_policy(loaded) must equal evaluate(path) (same seeds → identical metrics)."""
    m_path = grading.evaluate(MODEL_PATH, n_episodes=2)
    m_policy = grading.evaluate_policy(grading.load_policy(MODEL_PATH), n_episodes=2)
    assert m_path["median_reward"] == m_policy["median_reward"]
    assert m_path["success_rate"] == m_policy["success_rate"]


def test_contract_and_param_count(tiny_model):
    n = grading.check_contract(MODEL_PATH)
    assert 0 < n <= contract.PARAM_LIMIT


def test_grading_runs(tiny_model):
    m = grading.evaluate(MODEL_PATH, n_episodes=2)
    assert -contract.MAX_EPISODE_STEPS <= m["median_reward"] <= 0.0
    assert 0.0 <= m["success_rate"] <= 1.0


def test_latency_under_budget(tiny_model):
    assert grading.measure_latency(MODEL_PATH, n=20) <= contract.LATENCY_BUDGET_S


def test_frozen_module_rejected():
    """A jit.freeze'd model hides its parameters — the grader must reject it, not pass it."""
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(int(np.prod(contract.OBS_SHAPE)), contract.ACTION_DIM), torch.nn.Tanh())
    traced = torch.jit.freeze(torch.jit.trace(net.eval(), torch.zeros(1, *contract.OBS_SHAPE)))
    torch.jit.save(traced, MODEL_PATH + ".frozen")
    with pytest.raises(AssertionError):
        grading.count_parameters(MODEL_PATH + ".frozen")
    os.remove(MODEL_PATH + ".frozen")
