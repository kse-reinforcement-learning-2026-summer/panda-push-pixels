"""panda-push-pixels — frozen pixel Push environment + grader for KSE RL Project 2.

Quick start
-----------
    import gymnasium as gym
    import panda_push_pixels
    from panda_push_pixels import export_model, extract_actor, selfcheck, grading, contract
    from panda_push_pixels import render_episode

    env = gym.make("PandaPushPixels-v0")   # or panda_push_pixels.make_eval_env()
    render_episode(env)                    # sanity-check a rollout (random actions by default)
    # ... wrap with your own shaping, train an SB3 model ...
    export_model(model, "model.pt")        # SB3 -> standalone TorchScript (actor only)
    selfcheck(model, "model.pt")           # verify the export matches the trained policy
    print(grading.evaluate("model.pt"))    # the same metric the grader computes

    # or evaluate an already-loaded policy directly:
    import torch
    print(grading.evaluate_policy(torch.jit.load("model.pt")))
"""

import gymnasium as gym

from . import contract, grading
from .env import PandaPushPixels
from .export import export_model, extract_actor, selfcheck
from .viz import render_episode, save_video

__version__ = "9.0.0"

# Register with gymnasium so students can use gym.make("PandaPushPixels-v0")
gym.register(
    id="PandaPushPixels-v0",
    entry_point="panda_push_pixels.env:PandaPushPixels",
)


def make_eval_env():
    """Return the exact frozen environment the grader uses. Do not wrap it for evaluation."""
    return PandaPushPixels()


__all__ = [
    "PandaPushPixels",
    "make_eval_env",
    "export_model",
    "extract_actor",
    "selfcheck",
    "render_episode",
    "save_video",
    "grading",
    "contract",
    "__version__",
]
