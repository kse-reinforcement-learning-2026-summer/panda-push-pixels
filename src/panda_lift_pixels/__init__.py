"""panda-lift-pixels — frozen pixel Lift environment + grader for KSE RL Project 2.

Quick start
-----------
    import gymnasium as gym
    import panda_lift_pixels
    from panda_lift_pixels import export_model, extract_actor, selfcheck, grading, contract

    env = gym.make("PandaLiftPixels-v0")   # or panda_lift_pixels.make_eval_env()
    # ... wrap with your own shaping/curriculum, train an SB3 model ...
    export_model(model, "model.pt")        # SB3 -> standalone TorchScript (actor only)
    selfcheck(model, "model.pt")           # verify the export matches the trained policy
    print(grading.evaluate("model.pt"))    # the same metric the grader computes

    # or evaluate an already-loaded policy directly:
    import torch
    print(grading.evaluate_policy(torch.jit.load("model.pt")))
"""

import gymnasium as gym

from . import contract, grading
from .env import PandaLiftPixels
from .export import export_model, extract_actor, selfcheck

__version__ = "5.0.0"

# Register with gymnasium so students can use gym.make("PandaLiftPixels-v0")
gym.register(
    id="PandaLiftPixels-v0",
    entry_point="panda_lift_pixels.env:PandaLiftPixels",
)


def make_eval_env():
    """Return the exact frozen environment the grader uses. Do not wrap it for evaluation."""
    return PandaLiftPixels()


__all__ = [
    "PandaLiftPixels",
    "make_eval_env",
    "export_model",
    "extract_actor",
    "selfcheck",
    "grading",
    "contract",
    "__version__",
]
