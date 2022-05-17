import gym
import numpy as np
from gym import spaces


class SB3gym(gym.Env):
    def __init__(self, cfg: dict, is_eval: bool):
        super(SB3gym, self).__init__()
        self.env = gym.make(cfg["task"])
        self.action_space = self.env.action_space

        obs_img_sz = None
        if 'augmentation' in cfg['train']:
            aug_cfg = cfg['train']['augmentation']
            if 'translate' in aug_cfg and aug_cfg['translate']:
                obs_img_sz = aug_cfg['translate_sz']
            elif 'crop' in aug_cfg and aug_cfg['crop']:
                obs_img_sz = aug_cfg['crop_sz']
            else:
                obs_img_sz = cfg['screen_sz']
        else:
            obs_img_sz = cfg['screen_sz']

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, obs_img_sz, obs_img_sz), dtype=np.uint8
        )

        self.screen_sz = self.observation_space.shape[2] if is_eval else cfg['screen_sz']

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
