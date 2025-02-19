from collections import defaultdict, deque, namedtuple
import math
import random
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class ClinicAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs, randomize: bool = True) -> int:
        if randomize and np.random.random() < self.epsilon:
            valid_actions = self.env.get_valid_actions()
            return np.random.choice(valid_actions)
        else:
            return np.argmax(self.q_values[tuple(obs)])

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[tuple(obs)][action]
        )

        self.q_values[tuple(obs)][action] = (
            self.q_values[tuple(obs)][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ClinicDQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
        tau: float = 0.005,
        batch_size: int = 128,
        device="mps",
    ):
        self.env = env
        self.device = device
        self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(
            device
        )
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.fail_memory = ReplayMemory(10_000)
        self.success_memory = ReplayMemory(10_000)

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.training_error = []

    def get_action(self, obs, randomize: bool = True) -> int:
        valid_actions = self.env.get_valid_actions()
        if randomize and np.random.random() < self.epsilon:
            return torch.tensor(
                [[np.random.choice(valid_actions)]],
                device=self.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():
                valid_action_mask = torch.tensor(
                    [(i in valid_actions) for i in range(self.env.action_space.n)],
                    device=self.device,
                    dtype=torch.bool,
                ).unsqueeze(0)
                obs = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                return (
                    self.policy_net(obs)[valid_action_mask]
                    .unsqueeze(0)
                    .max(1)
                    .indices.view(1, 1)
                )

    def optimize_model(self):
        if len(self.fail_memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.training_error.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update(self, state, action, reward, terminated, next_state):
        """Update the Q-value of an action."""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        reward = torch.tensor([reward], device=self.device)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        self.memory.push(state, action, next_state, reward)

        self.optimize_model()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                self.tau * policy_net_state_dict[key]
                + (1 - self.tau) * target_net_state_dict[key]
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
