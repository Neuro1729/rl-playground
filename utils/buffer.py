import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Supports sampling batches for training.
    """

    def __init__(self, obs_shape, action_size, capacity=10000, batch_size=32, seed=42):
        self.capacity = capacity
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Buffers
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

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

