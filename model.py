import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # ---- TRAIN ----

    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./logs/",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )

    model.learn(total_timesteps=600_000)

    episode_rewards = env.get_episode_rewards()
    episode_lengths = env.get_episode_lengths()

    episode_data_df = pd.DataFrame({'Episode Rewards': episode_rewards, 'Episode Lengths': episode_lengths})

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward")
    plt.show()

    # Episode length
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length")
    plt.show()

    model.save("ppo_bipedalwalker")

    env.close()

