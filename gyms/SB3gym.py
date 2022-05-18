import gym
import numpy as np
from gym import spaces
from algo.utils import get_aug_screen_sz


class SB3gym(gym.Env):
    def __init__(self, cfg: dict, is_eval: bool):
        super(SB3gym, self).__init__()
        self.env = gym.make(cfg["task"])
        self.action_space = self.env.action_space

        self.aug_screen_sz = None
        self.screen_sz = cfg['screen_sz']
        self.aug_screen_sz = get_aug_screen_sz(cfg=cfg)

        if is_eval:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(3, self.aug_screen_sz, self.aug_screen_sz), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(3, self.screen_sz, self.screen_sz), dtype=np.uint8
            )

    def step(self, action):
        _, reward, done, info = self.env.step(action=action)
        rgb_obs = self.env.render(
            mode="rgb_array", width=self.screen_sz, height=self.screen_sz
        )
        rgb_obs = np.transpose(rgb_obs, (2, 0, 1))
        return rgb_obs, reward, done, info

    def reset(self):
        _ = self.env.reset()
        rgb_obs = self.env.render(
            mode="rgb_array", width=self.screen_sz, height=self.screen_sz
        )
        return np.transpose(rgb_obs, (2, 0, 1))

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def seed(self, seed: int):
        self.env.seed(seed)
        self.seed = seed

    def close(self):
        return self.env.close()
