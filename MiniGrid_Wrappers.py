from gymnasium.core import ObservationWrapper
import gymnasium as gym
import cv2
from gymnasium import spaces
from collections import deque
import numpy as np


class GrayImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        new_spaces = spaces.Box(
            low=0, high=255, shape=(80, 80, 1), dtype="uint8"
        )
        self.observation_space = new_spaces

    def observation(self, obs):
        obs["image"] = cv2.resize(obs["image"], (80, 80))
        obs["image"] = obs["image"][:, :, 0:1] * 0.2989 + obs["image"][:, :, 1:2] * 0.5870 + obs["image"][:, :, 2:3] * 0.1140
        return obs["image"]


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames= deque(maxlen=self.num_stack)
        new_spaces = spaces.Box(
            low=0, high=255, shape=(self.num_stack,80, 80, 1), dtype="uint8"
        )
        self.observation_space = new_spaces

    def observation(self):
        assert len(self.frames) == self.num_stack
        return np.array(list(self.frames))
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(), reward, terminated or truncated, info

    def reset(self, **kwargs):
        observation,_ = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()


class MaxStepWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = max_steps
    
    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.count += 1
        if self.count >= self.max_steps:
            return obs, reward, True, info
        else:
            return obs, reward, terminated, info
    
    def reset(self, **kwargs):
        self.count = 0
        return self.env.reset(**kwargs)