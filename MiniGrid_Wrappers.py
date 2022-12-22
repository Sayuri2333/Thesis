import math
from gymnasium.core import ObservationWrapper, Wrapper
import gymnasium as gym
import cv2
from gymnasium import spaces
from collections import deque
import numpy as np

from Atari_Warppers import RunningMeanStd


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
        self.frames = deque(maxlen=self.num_stack if self.num_stack > 0 else 1)
        new_spaces = spaces.Box(
            low=0, high=255, shape=(self.num_stack,80, 80, 1), dtype="uint8"
        )
        self.observation_space = new_spaces

    def observation(self):
        if self.num_stack > 0:
            assert len(self.frames) == self.num_stack
            return np.array(list(self.frames))
        elif self.num_stack == 0:
            return self.frames[0]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(), reward, terminated or truncated, info

    def reset(self, **kwargs):
        observation,_ = self.env.reset(**kwargs)
        if self.num_stack > 0:
            [self.frames.append(observation) for _ in range(self.num_stack)]
        else:
            self.frames.append(observation)
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


class NormalizeObsWrapper(gym.Wrapper):
    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
    
    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.normalize(np.array([obs]))[0]
        return obs
    
class StateBonus(Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 0.1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)