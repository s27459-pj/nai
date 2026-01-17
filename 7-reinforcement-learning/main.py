"""
See README.md for running instructions, examples and authors.
"""

import argparse
import pathlib
import random
from collections import deque
from typing import NamedTuple

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.types import FileLike

# Training parameters
# ===================
# How many episodes to train
EPISODES = 500
# How many frames to stack before saving to memory
FRAME_STACK_SIZE = 4
# How many frames to collect before starting training
WARMUP_STEPS = 10_000
# Train every N frames
TRAIN_FREQUENCY = 4

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class DQN(nn.Module):
    """Deep Q-Network with CNN for processing game frames"""

    def __init__(self, n_actions: int) -> None:
        super().__init__()

        # Input: 4 stacked frames of 84x84 grayscale images
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # After convolutions: 84x84 -> 7x7
        conv_out_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque[Transition](maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize to 84x84 and normalize"""

    frame = frame[34:194]  # Crop to game area
    frame = frame[::2, ::2]  # Downsample to 80x80

    # Add 2px padding on each side -> 84x84
    frame = np.pad(frame, ((2, 2), (2, 2)), mode="constant")

    # Normalize to [0, 1]
    return frame.astype(np.float32) / 255.0


class Agent:
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = ReplayBuffer(100_000)

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update_freq = 1000
        # How much training steps the agent has taken
        self.steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def train_step(self) -> None:
        """Perform one training step"""

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.from_numpy(states).to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        rewards_t = torch.from_numpy(rewards).to(device)
        next_states_t = torch.from_numpy(next_states).to(device)
        dones_t = torch.from_numpy(dones).to(device)

        # Compute current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # Compute loss and optimize
        loss = nn.functional.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, path: FileLike, episode: int) -> None:
        torch.save(
            {
                "episode": episode,
                "policy_net_state": self.policy_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_checkpoint(self, path: FileLike) -> int:
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps = checkpoint["steps"]
        self.epsilon = checkpoint["epsilon"]
        return checkpoint["episode"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-checkpoints", action="store_true", default=False)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path)
    parser.add_argument("--load-checkpoint", type=pathlib.Path)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    save_checkpoints: bool = args.save_checkpoints
    checkpoint_dir: pathlib.Path = args.checkpoint_dir or pathlib.Path("checkpoints")
    load_checkpoint: pathlib.Path | None = args.load_checkpoint
    verbose: bool = args.verbose

    gym.register_envs(ale_py)

    print(f"Using device: {device}")

    env = gym.make("PongNoFrameskip-v4", render_mode="human", obs_type="grayscale")
    n_actions = env.action_space.n  # type: ignore

    agent = Agent(n_actions)
    starting_episode = 0
    if load_checkpoint is not None:
        loaded_episode = agent.load_checkpoint(load_checkpoint)
        starting_episode = loaded_episode
        print(
            f"Loaded checkpoint {load_checkpoint} | "
            f"Episode: {loaded_episode} | "
            f"Steps: {agent.steps} | "
            f"Epsilon: {agent.epsilon}"
        )

    total_steps = 0

    # Run n + EPISODES times (depends if we loaded a checkpoint)
    for episode in range(starting_episode, starting_episode + EPISODES):
        observation, _ = env.reset()
        observation = preprocess_frame(observation)

        # Initialize frame stack with first frame repeated
        # We'll be pushing new frames to this deque in order to always have
        # a stack of the last FRAME_STACK_SIZE frames
        frame_stack = deque([observation] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
        state = np.array(frame_stack)

        episode_reward = 0.0
        episode_steps = 0
        while True:
            # Select and perform action
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_observation = preprocess_frame(next_observation)

            # Update frame stack
            frame_stack.append(next_observation)
            next_state = np.array(frame_stack)

            # Store transition to the agent's memory
            done = terminated or truncated
            transition = Transition(state, action, float(reward), next_state, done)
            agent.memory.push(transition)

            # Only train after warm-up period and every TRAIN_FREQUENCY frames
            if total_steps >= WARMUP_STEPS and total_steps % TRAIN_FREQUENCY == 0:
                agent.train_step()

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_steps += 1

            if verbose:
                print(
                    f"[{episode + 1:>3}] "
                    f"total_steps={total_steps} "
                    f"agent_steps={agent.steps} "
                    f"episode_steps={episode_steps} "
                    f"reward={episode_reward} "
                    f"epsilon={agent.epsilon:.3f}"
                )

            if done:
                break

        print(
            f"Episode {episode + 1}/{starting_episode + EPISODES} | "
            f"Score: {episode_reward:.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Memory: {len(agent.memory)} | "
            f"Episode steps: {episode_steps} | "
            f"Total steps: {total_steps}"
        )

        if save_checkpoints and total_steps >= WARMUP_STEPS:
            checkpoint_path = checkpoint_dir / f"episode_{episode + 1}.pt"
            agent.save_checkpoint(checkpoint_path, episode + 1)
            print(f"Saved checkpoint {checkpoint_path}")

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
