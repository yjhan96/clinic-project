from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from buffers import PrioritizedReplayBuffer, Transition

SAMPLE_SIZE = 50_000
MAX_ABS_LOSS = 2.0


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


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out = out + residual
        out = F.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        N = 256
        self.blocks = nn.Sequential(
            nn.Linear(n_observations, N),
            nn.ReLU(),
            ResidualBlock(N),
            ResidualBlock(N),
            ResidualBlock(N),
            nn.Linear(N, n_actions),
        )

    def forward(self, x):
        return F.tanh(self.blocks(x)) / 2.0 + 0.5


class ClinicDQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        n_iter: int,
        writer: SummaryWriter | None = None,
        discount_factor: float = 1.0,
        tau: float = 0.005,
        batch_size: int = 128,
        num_lookbacks=5,
        device="mps",
    ):
        self.env = env
        self.device = device
        input_len = gym.spaces.utils.flatten_space(
            env.nonterminal_normalized_observation_space
        ).shape[0]
        self.policy_net = DQN(input_len, env.action_space.n).to(device)
        self.target_net = DQN(input_len, env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.memory = PrioritizedReplayBuffer(800_000, batch_size)
        self.prev_states_and_rewards = deque([])

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.writer = writer
        self.num_lookbacks = num_lookbacks
        self.batch_idx = 0

    def get_action(self, obs, randomize: bool = True) -> int:
        valid_actions = self.env.get_valid_actions()
        if randomize and (np.random.random() < self.epsilon):
            return torch.tensor(
                [[np.random.choice(valid_actions)]],
                device=self.device,
                dtype=torch.long,
            )
        else:
            self.policy_net.eval()
            self.target_net.eval()
            obs_arr = gym.spaces.utils.flatten(
                self.env.nonterminal_normalized_observation_space,
                self.env.normalize_state(obs),
            )
            obs_tensor = torch.tensor(
                obs_arr, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            valid_actions_tensor = torch.tensor(
                [False for _ in range(self.env.action_space.n)],
                dtype=torch.bool,
                device=self.device,
            )
            valid_actions_tensor[list(valid_actions)] = True

            with torch.no_grad():
                predictions = self.policy_net(obs_tensor)
                masked_predictions = torch.where(valid_actions_tensor, predictions, 0.0)
                return masked_predictions.max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < SAMPLE_SIZE:
            return

        self.policy_net.train()
        self.target_net.train()

        transitions, indices, is_weights = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        non_final_valid_actions = torch.cat(
            [a.unsqueeze(0) for a in batch.valid_actions if a is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            target_predictions = self.target_net(non_final_next_states)
            masked_predictions = torch.where(
                non_final_valid_actions, target_predictions, -1.0
            )
            next_state_values[non_final_mask] = masked_predictions.max(1).values

        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        with torch.no_grad():
            abs_loss = torch.abs(
                state_action_values.squeeze(1) - expected_state_action_values
            )
            abs_loss_clamped = torch.clamp(abs_loss, max=MAX_ABS_LOSS)
            self.memory.update_p_weights(list(zip(indices, abs_loss_clamped.tolist())))

        is_weights_tensor = torch.tensor(
            is_weights, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        criterion = nn.SmoothL1Loss(reduction="none")
        loss = (
            criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            * is_weights_tensor
        ).mean()

        if self.writer is not None:
            self.writer.add_scalar("Training Error", loss.item(), self.batch_idx)
            self.writer.add_scalar(
                "Sample mean reward", reward_batch.mean().item(), self.batch_idx
            )
        self.batch_idx += 1

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def _to_tensor(self, state, action, reward, terminated, next_state) -> tuple:
        state_arr = gym.spaces.utils.flatten(
            self.env.nonterminal_normalized_observation_space,
            self.env.normalize_state(state),
        )
        state_tensor = torch.tensor(
            state_arr, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        if terminated:
            next_state_tensor = None
            valid_actions_tensor = None
        else:
            next_state_arr = gym.spaces.utils.flatten(
                self.env.nonterminal_normalized_observation_space,
                self.env.normalize_state(next_state),
            )
            next_state_tensor = torch.tensor(
                next_state_arr,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            valid_action_indices = self.env.get_valid_actions()
            valid_actions_tensor = torch.tensor(
                [False for _ in range(self.env.action_space.n)],
                dtype=torch.bool,
                device=self.device,
            )
            valid_actions_tensor[list(valid_action_indices)] = True

        return (state_tensor, action, reward, next_state_tensor, valid_actions_tensor)

    def update(self, state, action, reward, terminated, next_state):
        """Update the Q-value of an action."""
        (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            valid_actions_tensor,
        ) = self._to_tensor(state, action, reward, terminated, next_state)
        self.prev_states_and_rewards.append((state_tensor, reward_tensor))
        while (terminated and len(self.prev_states_and_rewards) > 0) or len(
            self.prev_states_and_rewards
        ) == self.num_lookbacks:
            discount_factors = torch.tensor(
                [
                    self.discount_factor**i
                    for i in range(len(self.prev_states_and_rewards))
                ],
                dtype=torch.float32,
                device=self.device,
            )
            total_reward_tensor = (
                torch.mul(
                    torch.concat(
                        [
                            state_and_reward[1]
                            for state_and_reward in self.prev_states_and_rewards
                        ]
                    ),
                    discount_factors,
                )
                .sum()
                .unsqueeze(0)
            )
            state_tensor = self.prev_states_and_rewards.popleft()[0]
            self.memory.push(
                state_tensor,
                action_tensor,
                next_state_tensor,
                valid_actions_tensor,
                total_reward_tensor,
                p_weight=MAX_ABS_LOSS,
            )

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

    def update_lr(self):
        self.scheduler.step()
