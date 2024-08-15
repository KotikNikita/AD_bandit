# import numpy as np
from bandit import TaskDistribution, MultiArmedBandit
from agents import Agent
from tqdm import tqdm
import numpy as np


def train_n_episodes(env: MultiArmedBandit, agent: Agent, n: int, RTG: bool = False):
    observations = []
    actions = []
    rewards = []

    for _ in range(n):
        action = agent.act()
        reward = env.run(action)
        agent.update_policy(action, reward)
        observations.append(0)
        actions.append(action)
        rewards.append(reward)
    if RTG:
        tmp = rewards[::-1]
        tmp = np.array(tmp).cumsum()
        tmp = list(tmp)
        rewards = tmp[::-1]
    return {"observations": observations, "actions": actions, "rewards": rewards}


def generate_dataset(env_distr: TaskDistribution, agent: Agent, n: int, size: int, RTG: bool = False):
    dataset = []
    for _ in tqdm(range(size)):
        env = env_distr.sample_task()
        dataset.append(train_n_episodes(env, agent, n, RTG))
    return dataset
