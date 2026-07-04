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


def test_curriculum_start_grasped_lifts_object():
    env = make_eval_env()
    _, info = env.reset(seed=1, options={"start_grasped": True})
    assert info["object_height"] > contract.GRASP_LIFT_OFF
    env.close()


def test_is_touching_present_and_pure_contact():
    """info exposes a pure-contact is_touching signal (grasp without the height gate)."""
    env = make_eval_env()
    _, info = env.reset(seed=0)
    # both keys present, boolean-typed
    assert "is_touching" in info and "is_grasped" in info
    assert isinstance(bool(info["is_touching"]), bool)
    # is_grasped implies is_touching (grasped = touching AND lifted), never the reverse constraint
    for _ in range(20):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert not (info["is_grasped"] and not info["is_touching"])
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
