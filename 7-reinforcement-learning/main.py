"""
See README.md for running instructions, examples and authors.
"""

from enum import Enum

import ale_py
import gymnasium as gym
import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Action(int, Enum):
    """
    Action space for the Pong environment

    https://ale.farama.org/environments/pong/#actions
    """

    NOOP = 0
    """No operation, do nothing"""
    FIRE = 1
    """Press the fire button without updating the joystick position"""
    RIGHT = 2
    """Apply a Δ-movement rightward on the joystick"""
    LEFT = 3
    """Apply a Δ-movement leftward on the joystick"""
    RIGHTFIRE = 4
    """Execute RIGHT and FIRE   """
    LEFTFIRE = 5
    """Execute LEFT and FIRE"""


def main() -> None:
    gym.register_envs(ale_py)

    # https://ale.farama.org/environments/pong/
    env = gym.make("PongNoFrameskip-v4", render_mode="human", obs_type="grayscale")
    n_actions = env.action_space.n  # type: ignore

    observation, info = env.reset(seed=42)
    n_observations = len(observation)

    score = 0.0
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        score += float(reward)

        print(f"{action=} {reward=} {score=} {info=}")

        if terminated or truncated:
            print(f"Your final score is {score}")
            break

    env.close()


if __name__ == "__main__":
    main()
