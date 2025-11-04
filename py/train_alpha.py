# --- at top of file ---
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from envs.gym_alpha_env import GymAlphaEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(csv_path, window=32, fee_bps=1.0):
    def _f():
        # Monitor writes episodic stats to CSV for convergence plots
        return Monitor(GymAlphaEnv(csv_path, window, fee_bps),
                       filename=os.path.join(LOG_DIR, "monitor.csv"))
    return _f

if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "data", "sample_btc_minute.csv")
    env = DummyVecEnv([make_env(csv)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,        # <— SB3 will emit TB scalars
    )

    # Also dump CSV logs alongside TB
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=500_000)
    model.save("ppo_alpha.zip")
    env.save("vecnorm.pkl")
    print("Saved model and normalization stats.")
