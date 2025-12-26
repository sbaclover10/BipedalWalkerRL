from envs.wrappers import ObservationNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def make_env(seed=42):
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(8)])

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        n_steps=2048,
        batch_size=64,
        n_epochs=13,
        learning_rate=0.0004,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="../logs/"
    )

    model.learn(total_timesteps=800_000, tb_log_name="OBSERVATION_NORM")
    model.save("logs/OBSERVATION_NORM/")
