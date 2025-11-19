import torch
import numpy as np
from wrapper import GymWrapper
from models.actor import Actor
from training.config import Config

def evaluate(cfg: Config, actor_path: str, render: bool = True):
    # Initialize environment
    env = GymWrapper(cfg.env_name, cfg).get_env()
    
    # Initialize actor network
    obs_dim = cfg.obs_shape[0]
    actor = Actor(obs_dim, cfg.action_size,
                  hidden_size=cfg.hidden_size,
                  num_layers=cfg.actor_layers,
                  activation=cfg.activation)
    
    # Load saved actor weights
    actor.load_state_dict(torch.load(actor_path))
    actor.eval()
    
    total_rewards = []

    for ep in range(cfg.eval_episodes):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        ep_reward = 0

        while not done:
            with torch.no_grad():
                probs = actor(obs)
            action = torch.argmax(probs).item()
            obs, reward, done, _ = env.step(action)
            obs = torch.tensor(obs, dtype=torch.float32)
            ep_reward += reward

            if render:
                env.render()

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {cfg.eval_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    cfg = Config()
    actor_path = "results/models/actor_ep500.pt"  # update path if needed
    evaluate(cfg, actor_path, render=True)
