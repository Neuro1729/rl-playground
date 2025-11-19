import gymnasium as gym
import numpy as np
from gymnasium import spaces

class OneHotAction(gym.ActionWrapper):
    """
    Converts a discrete action into a one-hot vector.
    Useful for actor networks that output a probability distribution.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), \
            "OneHotAction wrapper only works with discrete action spaces."
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32)

    def action(self, action):
        """Convert one-hot vector to integer index for env.step"""
        index = np.argmax(action).astype(int)
        return index

    def reverse_action(self, action):
        """Convert integer index back to one-hot vector (for evaluation if needed)"""
        vec = np.zeros(self.action_space.shape, dtype=np.float32)
        vec[action] = 1.0
        return vec

class ActionRepeat(gym.Wrapper):
    """Repeats the same action for 'repeat' steps."""
    def __init__(self, env, repeat=1):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    """Ends episode after max_steps."""
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        done = terminated or truncated
        if self.current_step >= self.max_steps:
            done = True
            info["time_limit_reached"] = True
        return obs, reward, done, info

class GymWrapper:
    """Factory to create Gymnasium env with optional wrappers."""
    def __init__(self, env_name, cfg):
        self.cfg = cfg
        self.env = gym.make(env_name)

        # Apply wrappers
        if cfg.use_pixel:
            raise NotImplementedError("Pixel-based wrapper not implemented yet")
        if cfg.action_repeat > 1:
            self.env = ActionRepeat(self.env, repeat=cfg.action_repeat)
        self.env = TimeLimit(self.env, cfg.max_steps)
        self.env = OneHotAction(self.env)

    def get_env(self):
        return self.env
