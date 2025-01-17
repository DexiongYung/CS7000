import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from algo.envs import VecPyTorch, make_vec_envs
from algo.utils import get_render_func, get_vec_normalize

import envs

sys.path.append('algo')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='HalfCheetah-v2',
    help='environment that is trained on (default: HalfCheetah-v2)')
parser.add_argument(
    '--load-dir',
    default=None,
    help='directory of saved agent logs (default: None)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--random',
    action='store_true',
    default=False,
    help='whether to execute random actions when the learned policy is not provided'
    )
parser.add_argument(
    '--still',
    action='store_true',
    default=False,
    help='whether to just stay still without any action'
)
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 3000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()

# We need to use the same statistics for normalization as used in training
if args.load_dir is None:
    actor_critic = None
else:
    load_path = args.load_dir if args.load_dir.endswith('.pt') else os.path.join(args.load_dir, args.env_name + '.pt')
    actor_critic, ob_rms = torch.load(load_path)
    
    if vec_norm is not None:
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(1,
                                        actor_critic.recurrent_hidden_state_size)

masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

episode_reward = 0
episode_length = 0

if args.still:
    while True:
        render_func('human')
else:
    while True:
        if actor_critic is None:
            if args.random:
                action = torch.tensor(env.action_space.sample())
            else:
                action = torch.zeros(env.action_space.shape)
        else:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        episode_reward += reward.numpy()[0][0]
        episode_length += 1

        if done:
            print(f'Episode reward {episode_reward}, length {episode_length}')
            episode_reward = 0
            episode_length = 0

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')
