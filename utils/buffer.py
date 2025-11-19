# utils/buffer.py
import numpy as np
import random
from training.config import Config

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Uses hyperparameters from Config.
    Supports sampling batches for training.
    """

    def __init__(self, cfg: Config):
        self.capacity = cfg.buffer_capacity
        self.batch_size = cfg.batch_size
        self.obs_shape = cfg.obs_shape
        self.action_size = cfg.action_size
        self.seed = cfg.seed

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Buffers
        self.states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_size), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        """Randomly sample a batch of experiences."""
        idx = np.random.choice(self.size, self.batch_size, replace=False)

        batch = dict(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx]
        )
        return batch

    def __len__(self):
        return self.size
