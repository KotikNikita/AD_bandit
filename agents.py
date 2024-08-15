import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, k: int = 10) -> None:
        self.k = k

    def act(self):
        return np.random.randint(0, self.k)


class EpsilonGreedy(Agent):
    def __init__(self, k: int = 10,
                 epsilon: float = 0.1) -> None:
        super().__init__(k)
        self.epsilon = epsilon
        self.rewards = [None] * k
        self.states_counter = [None] * k
        self.explored = 0

    def first_exploration(self):
        self.explored += 1
        return self.explored - 1

    def first_updates(self, action, reward):
        self.rewards[action] = reward
        self.states_counter[action] = 1

    def update_policy(self, action, reward):
        if self.explored == self.k + 1:
            self.rewards[action] = (self.rewards[action] * self.states_counter[action] + reward)
            self.states_counter[action] += 1
            self.rewards[action] /= self.states_counter[action]
        else:
            self.first_updates(action, reward)

    def act(self):
        if self.explored == self.k:
            p = np.random.uniform(0, 1)
            if p < self.epsilon:
                return np.random.randint(0, self.k)
            else:
                return np.argmax(self.rewards)
        else:
            return self.first_exploration()


class UCB_Agent(Agent):
    def __init__(self, k: int = 10,
                 c: float = 0.1) -> None:
        super().__init__(k)
        self.c = c
        self.rewards = [None] * k
        self.states_counter = [None] * k
        self.explored = 0

    def first_exploration(self):
        # self.explored += 1
        return self.explored

    def first_updates(self, action, reward):
        self.rewards[action] = reward
        self.states_counter[action] = 1

    def update_policy(self, action, reward):
        if self.explored == self.k:
            self.rewards[action] = (self.rewards[action] * self.states_counter[action] + reward)
            self.states_counter[action] += 1
            self.rewards[action] /= self.states_counter[action]
        else:
            self.first_updates(action, reward)
            self.explored += 1

    def print_pseudo_rewards(self):
        print('pseudo_rewards =' +
              f'{self.rewards + self.c * np.sqrt(np.log(sum(self.states_counter)) / self.states_counter)}')
        print(f'self.rewards = {self.rewards}')
        print(f'self.states_counter = {self.states_counter}')

    def plot_pseudo_rewards(self):
        plt.scatter(np.arange(self.k), self.rewards, label="mean rewards")
        plt.scatter(np.arange(self.k),
                    self.rewards + self.c * np.sqrt(np.log(sum(self.states_counter)) / self.states_counter),
                    label="UCB of the rewards")
        plt.legend()
        plt.grid(True)
        plt.show()

    def act(self):
        if self.explored == self.k:
            pseudo_rewards = self.rewards + self.c * np.sqrt(np.log(sum(self.states_counter)) / self.states_counter)
            return np.argmax(pseudo_rewards)
        else:
            return self.first_exploration()
