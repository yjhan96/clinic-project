import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clinic_agent import ClinicDQNAgent
from clinic_environment import ClinicEnv

NUM_EPISODES = 40_000


def make_agents(clinic_args, device) -> list:
    return [
        ClinicDQNAgent(
            ClinicEnv(**clinic_args),
            learning_rate=1e-3,
            initial_epsilon=1.0,
            epsilon_decay=1.0 / (NUM_EPISODES / 2.0),
            final_epsilon=0.1,
            n_iter=NUM_EPISODES,
            batch_size=128,
            device=device,
            writer=SummaryWriter(
                comment="3_NURSE_DQN_PRIORITIZED_REPLAY_LR_1e-3_BS_128"
            ),
        ),
        ClinicDQNAgent(
            ClinicEnv(**clinic_args),
            learning_rate=1e-4,
            initial_epsilon=1.0,
            epsilon_decay=1.0 / (NUM_EPISODES / 2.0),
            final_epsilon=0.1,
            n_iter=NUM_EPISODES,
            batch_size=128,
            device=device,
            writer=SummaryWriter(
                comment="3_NURSE_DQN_PRIORITIZED_REPLAY_LR_1e-4_BS_128"
            ),
        ),
        ClinicDQNAgent(
            ClinicEnv(**clinic_args),
            learning_rate=1e-3,
            initial_epsilon=1.0,
            epsilon_decay=1.0 / (NUM_EPISODES / 2.0),
            final_epsilon=0.1,
            n_iter=NUM_EPISODES,
            batch_size=256,
            device=device,
            writer=SummaryWriter(
                comment="3_NURSE_DQN_PRIORITIZED_REPLAY_LR_1e-3_BS_256"
            ),
        ),
        ClinicDQNAgent(
            ClinicEnv(**clinic_args),
            learning_rate=1e-4,
            initial_epsilon=1.0,
            epsilon_decay=1.0 / (NUM_EPISODES / 2.0),
            final_epsilon=0.1,
            n_iter=NUM_EPISODES,
            batch_size=256,
            device=device,
            writer=SummaryWriter(
                comment="3_NURSE_DQN_PRIORITIZED_REPLAY_LR_1e-4_BS_256"
            ),
        ),
    ]


def play_episode(env, agent, randomize: bool = True, update_model: bool = True):
    obs, info = env.reset()
    done = False

    total_reward = 0
    while not done:
        action = agent.get_action(obs, randomize=randomize)
        next_obs, reward, terminated, truncated, info = env.step(action.item())

        if update_model:
            agent.update(obs, action, reward, terminated, next_obs)
        else:
            print(action.item())

        done = terminated or truncated
        obs = next_obs
        total_reward += reward

    return total_reward


def train_agent(agent, n_episodes):
    for i in range(n_episodes):
        total_reward = play_episode(agent.env, agent)
        agent.writer.add_scalar("total reward", total_reward, i)


def train_agents(agents, n_episodes):
    for i in tqdm(range(n_episodes)):
        for agent in agents:
            total_reward = play_episode(agent.env, agent)
            agent.writer.add_scalar("total reward", total_reward, i)


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    clinic_args = {
        "clinic_capacity": np.array([1, 2]),
        "clinic_travel_times": np.array([[0, 10], [10, 0]]),
        "patient_times": np.array([30, 40, 50]),
        "num_nurses": 3,
    }
    agents = make_agents(clinic_args, device)
    train_agents(agents, NUM_EPISODES)


if __name__ == "__main__":
    main()
