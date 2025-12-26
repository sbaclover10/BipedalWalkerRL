import gymnasium as gym
import numpy as np

class ObservationNorm(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.running_mean = np.zeros(env.observation_space.shape)
        self.running_var = np.ones(env.observation_space.shape)
        self.count = 1e-4

    def observation(self, obs):
        self.count += 1
        self.running_mean += (obs - self.running_mean) / self.count
        self.running_var += (obs - self.running_mean) ** 2
        return (obs - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
