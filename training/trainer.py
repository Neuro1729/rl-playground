import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import os

from models.actor import Actor
from models.critic import Critic
from utils.buffer import ReplayBuffer
from wrapper import GymWrapper

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # --- Environment ---
        self.env = GymWrapper(cfg.env_name, cfg).get_env()
        self.obs_dim = int(np.prod(cfg.obs_shape))  # handle vector obs
        self.action_size = cfg.action_size

        # --- Networks ---
        self.actor = Actor(
            self.obs_dim, self.action_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.actor_layers,
            activation=cfg.activation
        )
        self.critic = Critic(
            self.obs_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.critic_layers,
            activation=cfg.activation
        )

        # --- Optimizers ---
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(cfg)

        # --- Logging / Directories ---
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        self.gamma = cfg.gamma
        self.lambda_ = cfg.lambda_

    def compute_gae(self, rewards, values, masks):
        """Compute Generalized Advantage Estimation (GAE)"""
        values = values + [0]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lambda_ * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def train(self):
        episode_rewards = deque(maxlen=100)

        for ep in range(1, self.cfg.train_episodes + 1):
            # --- Reset environment (Gymnasium returns obs, info) ---
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            done = False
            total_reward = 0

            rewards = []
            values = []
            log_probs = []
            masks = []

            while not done:
                # --- Actor selects action ---
                probs = self.actor(obs)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.critic(obs)

                # --- Step environment (Gymnasium) ---
                next_obs, reward, terminated, truncated, info = self.env.step(action.numpy())
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32)

                # --- Store trajectory ---
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob)
                masks.append(1 - float(done))

                obs = next_obs
                total_reward += reward

            # --- Compute returns and advantages ---
            returns = self.compute_gae(rewards, values, masks)
            returns = torch.tensor(returns, dtype=torch.float32)
            log_probs = torch.stack(log_probs)
            values = torch.tensor(values, dtype=torch.float32)

            advantages = returns - values

            # --- Actor loss ---
            actor_loss = -(log_probs * advantages.detach()).mean()

            # --- Critic loss ---
            critic_loss = F.mse_loss(values, returns)

            # --- Update networks ---
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
            self.critic_opt.step()

            episode_rewards.append(total_reward)

            # --- Logging ---
            if ep % 10 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {ep} | Avg Reward: {avg_reward:.2f}")

            # --- Save models ---
            if ep % self.cfg.save_every == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.cfg.model_dir, f"actor_ep{ep}.pt"))
                torch.save(self.critic.state_dict(), os.path.join(self.cfg.model_dir, f"critic_ep{ep}.pt"))
