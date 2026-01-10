"""
See README.md for running instructions, examples and authors.
"""

from enum import Enum

import ale_py
import gymnasium as gym


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
    env = gym.make("ALE/Pong-v5", render_mode="human")

    observation, info = env.reset(seed=42)

    for i in range(1000):
        # TODO: Select action using an agent (reinforcement learning)
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"[{i:>2}] {action=} {reward=} {info=}")

        if terminated or truncated:
            print("Terminated or truncated -> reset")
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
