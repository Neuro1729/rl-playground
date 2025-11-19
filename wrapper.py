import gymnasium as gym
import numpy as np
from gymnasium import spaces

class OneHotAction(gym.Wrapper):
    """Wrap discrete actions into one-hot vectors"""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32)

    def action(self, action):
        return int(np.argmax(action))

    def reverse_action(self, action):
        vec = np.zeros(self.action_space.shape, dtype=np.float32)
        vec[action] = 1.0
        return vec

class ActionRepeat(gym.Wrapper):
    """Repeat same action for N steps"""
    def __init__(self, env, repeat=1):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, term, trunc, i = self.env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc
            info.update(i)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class TimeLimit(gym.Wrapper):
    """End episode after max_steps"""
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
        if self.current_step >= self.max_steps:
            truncated = True
            info["time_limit_reached"] = True
        return obs, reward, terminated, truncated, info

class GymWrapper:
    """Factory to create Gymnasium env with wrappers"""
    def __init__(self, env_name, cfg):
        self.cfg = cfg
        self.env = gym.make(env_name, render_mode="human" if cfg.render else None)

        if cfg.use_pixel:
            raise NotImplementedError("Pixel-based wrapper not implemented")
        if cfg.action_repeat > 1:
            self.env = ActionRepeat(self.env, repeat=cfg.action_repeat)
        self.env = TimeLimit(self.env, cfg.max_steps)
        self.env = OneHotAction(self.env)

    def get_env(self):
        return self.env
