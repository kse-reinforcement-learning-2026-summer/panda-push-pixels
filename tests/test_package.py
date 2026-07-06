"""Self-tests for the panda-push-pixels package (run in the package repo's own CI).

These verify the frozen contract and the grading pipeline. They train a tiny model on CPU, so
they are slow-ish but self-contained. Requires the ``[train]`` extra (Stable-Baselines3).
"""

import os

import gymnasium as gym
import numpy as np
import pytest
import torch

import panda_push_pixels
from panda_push_pixels import contract, export_model, make_eval_env, selfcheck
from panda_push_pixels import grading

MODEL_PATH = "/tmp/_ppp_selftest_model.pt"


def test_env_observation_contract():
    env = make_eval_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == contract.OBS_SHAPE
    assert obs.dtype == np.uint8
    assert obs.min() >= 0 and obs.max() <= 255
    assert tuple(env.action_space.shape) == (contract.ACTION_DIM,)
    assert {
        "object_position", "target_position", "object_size", "ee_position",
        "object_to_target", "ee_to_object", "is_touching", "is_success",
    } <= set(info)
    env.close()


def test_episode_horizon_and_reward():
    """Horizon is MAX_EPISODE_STEPS unless the dwell condition ends it sooner."""
    env = make_eval_env()
    env.reset(seed=0)
    rewards, done, steps = [], False, 0
    info = None
    while not done:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        rewards.append(r)
        steps += 1
        done = term or trunc
    assert steps <= contract.MAX_EPISODE_STEPS
    assert set(np.unique(rewards)).issubset({contract.STEP_PENALTY, contract.STEP_PENALTY + contract.SUCCESS_BONUS})
    if steps < contract.MAX_EPISODE_STEPS:
        assert info["is_success"]        # the only way to end before the horizon is dwell success
    env.close()


def test_never_spawns_already_solved():
    """reset() must never hand out a configuration whose cube is already within the success zone --
    otherwise a do-nothing policy scores a freebie success, corrupting shaping and the graded metric."""
    env = make_eval_env()
    for seed in range(60):
        _, info = env.reset(seed=seed)
        assert info["object_to_target"] >= contract.DISTANCE_THRESHOLD
        assert info["is_success"] is False
    env.close()


def test_dwell_success_terminates_with_bonus():
    """The cube must stay within the target threshold for DWELL_STEPS consecutive steps -- a
    single-step graze from a fast-moving cube must NOT be enough (that was the old bug)."""
    env = make_eval_env()
    env.reset(seed=0)
    tgt = env._sim.get_base_position("target")
    env._sim.set_base_pose("object", tgt, np.array([0.0, 0.0, 0.0, 1.0]))  # teleport cube onto target
    zero = np.zeros(contract.ACTION_DIM, dtype=np.float32)
    for _ in range(contract.DWELL_STEPS - 1):
        _, reward, terminated, truncated, info = env.step(zero)
        assert terminated is False
        assert info["is_success"] is False
        assert reward == contract.STEP_PENALTY
    _, reward, terminated, truncated, info = env.step(zero)   # the DWELL_STEPS-th consecutive close step
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert reward == contract.STEP_PENALTY + contract.SUCCESS_BONUS
    env.close()


def test_timeout_with_cube_at_target_still_counts_as_success():
    """If the cube reaches the target only right at the time limit, that still counts -- it would
    have dwelled long enough given a few more steps; the horizon shouldn't punish a near-miss."""
    env = make_eval_env()
    env.reset(seed=0)   # seed 0's object/target spawn ~0.11m apart -- not already close
    zero = np.zeros(contract.ACTION_DIM, dtype=np.float32)
    for _ in range(contract.MAX_EPISODE_STEPS - 1):
        _, _, terminated, truncated, info = env.step(zero)
        assert not terminated and not truncated
        assert info["is_success"] is False
    tgt = env._sim.get_base_position("target")
    env._sim.set_base_pose("object", tgt, np.array([0.0, 0.0, 0.0, 1.0]))  # teleport onto target for the LAST step
    _, reward, terminated, truncated, info = env.step(zero)
    assert terminated is False
    assert truncated is True
    assert info["is_success"] is True
    assert reward == contract.STEP_PENALTY + contract.SUCCESS_BONUS
    env.close()


def test_privileged_info_typed_and_consistent():
    """info exposes what's needed to build a reward: positions/size/contact, correctly typed."""
    env = make_eval_env()
    _, info = env.reset(seed=0)
    assert isinstance(bool(info["is_touching"]), bool)
    assert info["object_position"].shape == (3,)
    assert info["target_position"].shape == (3,)
    assert info["ee_position"].shape == (3,)
    assert info["object_size"] == contract.OBJECT_SIZE
    assert info["object_to_target"] == pytest.approx(
        float(np.linalg.norm(info["object_position"] - info["target_position"])), abs=1e-5
    )
    for _ in range(20):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if info["is_success"]:           # success => currently close (the converse need not hold: it
            assert info["object_to_target"] < contract.DISTANCE_THRESHOLD  # may take DWELL_STEPS to fire)
        if term or trunc:
            break
    env.close()


@pytest.fixture(scope="module")
def tiny_model():
    from stable_baselines3 import SAC

    env = gym.make("PandaPushPixels-v0", max_episode_steps=50)
    model = SAC(
        "CnnPolicy", env, buffer_size=400, learning_starts=40, train_freq=4, batch_size=32,
        device="cpu", verbose=0,   # default normalize_images=True: SB3 /255 on the uint8 obs
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
    from panda_push_pixels import extract_actor

    Algo = getattr(sb3, algo_name)
    env = gym.make("PandaPushPixels-v0")
    kwargs = dict(device="cpu", verbose=0)   # default normalize_images=True on the uint8 obs
    if algo_name in ("DDPG", "TD3", "SAC"):
        kwargs["buffer_size"] = 200  # tiny: avoid allocating a huge image replay buffer
    model = Algo("CnnPolicy", env, **kwargs)

    actor = extract_actor(model).eval()
    obs_batch = np.random.randint(0, 256, (4, *contract.OBS_SHAPE), dtype=np.uint8)
    with torch.no_grad():
        actions_actor = actor(torch.as_tensor(obs_batch, dtype=torch.float32)).numpy()
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
    lower = contract.STEP_PENALTY * contract.MAX_EPISODE_STEPS               # total failure
    upper = contract.STEP_PENALTY * contract.DWELL_STEPS + contract.SUCCESS_BONUS  # fastest possible success
    assert lower <= m["median_reward"] <= upper
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
