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
    env = make_eval_env()
    env.reset(seed=0)
    rewards, done, steps = [], False, 0
    while not done:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        rewards.append(r)
        steps += 1
        done = term or trunc
    assert steps == contract.MAX_EPISODE_STEPS
    assert set(np.unique(rewards)).issubset({-1.0, 0.0})
    env.close()


def test_curriculum_start_grasped_lifts_object():
    env = make_eval_env()
    _, info = env.reset(seed=1, options={"start_grasped": True})
    assert info["object_height"] > contract.GRASP_LIFT_OFF
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
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12 * 96 * 96, 8), torch.nn.Tanh())
    traced = torch.jit.freeze(torch.jit.trace(net.eval(), torch.zeros(1, *contract.OBS_SHAPE)))
    torch.jit.save(traced, MODEL_PATH + ".frozen")
    with pytest.raises(AssertionError):
        grading.count_parameters(MODEL_PATH + ".frozen")
    os.remove(MODEL_PATH + ".frozen")
