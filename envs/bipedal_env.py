import gymnasium as gym
import numpy as np

class ShapedBipedalWalker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # reward shaping
        forward_vel = obs[2]
        energy_penalty = 0.001 * np.sum(np.square(action))

        shaped_reward = reward + 0.5 * forward_vel - energy_penalty # not too much reward shaping...

        return obs, shaped_reward, terminated, truncated, info
