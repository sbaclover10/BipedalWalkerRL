import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(seed=42):
    def _init():
        env = gym.make("BipedalWalker-v3")
        env.reset(seed=seed)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        n_steps=2048,
        n_epochs=13,
        batch_size=64,
        learning_rate=0.0004,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="../logs/"
    )

    model.learn(
        total_timesteps=800_000,
        tb_log_name="HIGH_ENTROPY"
    )

    model.save("../models/HIGH_ENTROPY/")
