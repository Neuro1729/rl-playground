from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class Config:
    # --- Environment ---
    env_name: str = "CartPole-v1"   # Gym environment
    obs_shape: Tuple = (4,)          # observation shape
    action_size: int = 2             # number of discrete actions
    max_steps: int = 200             # max steps per episode
    seed: int = 42                   # random seed

    # --- Training ---
    train_episodes: int = 500        # total episodes for training
    eval_episodes: int = 10          # episodes to evaluate
    gamma: float = 0.99              # discount factor
    lambda_: float = 0.95            # GAE lambda
    horizon: int = 10                 # rollout horizon for value estimation
    batch_size: int = 32
    update_every: int = 1             # update model every N steps
    grad_clip: float = 1.0            # clip gradients

    # --- Actor-Critic ---
    hidden_size: int = 128
    actor_layers: int = 2
    critic_layers: int = 2
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    actor_entropy_scale: float = 0.01  # for exploration
    activation: str = "ReLU"           # ReLU, ELU, Tanh

    # --- Replay Buffer ---
    buffer_capacity: int = 10000
    prioritized_replay: bool = False   # can be extended later

    # --- Exploration ---
    epsilon_start: float = 1.0         # for epsilon-greedy
    epsilon_end: float = 0.05
    epsilon_decay: int = 5000          # steps to decay

    # --- Pixel / Vision (optional for future) ---
    use_pixel: bool = False
    action_repeat: int = 1
    frame_stack: int = 4

    # --- Logging & Saving ---
    save_every: int = 50               # episodes
    render: bool = False
    log_dir: str = "results/logs"
    model_dir: str = "results/models"
    gif_dir: str = "results/gifs"

    # --- Advanced / RL-specific ---
    use_slow_target: bool = True       # for target networks
    slow_target_update: int = 100
    slow_target_fraction: float = 0.99
