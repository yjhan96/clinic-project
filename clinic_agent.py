from collections import defaultdict
import math
import gymnasium as gym
import numpy as np


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
        num_total_actions = math.prod([space.n for space in env.action_space.spaces])
        self.q_values = defaultdict(lambda: np.zeros(num_total_actions))
        self.num_action_per_nurse: int = env.action_space.spaces[0].n
        self.num_nurses: int = len(env.action_space.spaces)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def _to_action(self, action_int: int) -> tuple[int]:
        res = []
        curr_num = action_int
        curr_exp = self.num_nurses - 1
        while curr_exp > 0:
            divider = self.num_action_per_nurse**curr_exp
            quotient = curr_num // divider
            res.append(quotient)
            curr_num -= quotient * divider
            curr_exp -= 1
        res.append(curr_num)
        return tuple(res)

    def _from_action(self, action: tuple[int]) -> int:
        res = 0
        for nurse_action, exp in zip(
            action, range(self.num_nurses - 1, -1, -1), strict=True
        ):
            res += nurse_action * (self.num_action_per_nurse**exp)
        return res

    def get_action(self, obs, randomize: bool = True) -> tuple[int]:
        if randomize and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            action_int = np.argmax(self.q_values[tuple(obs)])
            return self._to_action(action_int)

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates the Q-value of an action."""
        action_int = self._from_action(action)
        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[tuple(obs)][action_int]
        )

        self.q_values[tuple(obs)][action_int] = (
            self.q_values[tuple(obs)][action_int] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
