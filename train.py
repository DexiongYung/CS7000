import os
import time
import json
import datetime
from collections import deque

import numpy as np
import torch

from algo import PPO, utils
from algo.model import Policy, AugPolicy
from algo.storage import RolloutStorage
from algo.utils import get_config, get_logger, save_model
from evaluation import evaluate
from env import get_env
from torch.utils.tensorboard import SummaryWriter
from augmentations.Augmenter import Augmenter


def main(cfg: dict):
    task = cfg['task']
    seed = cfg['seed']
    num_workers = cfg['num_workers']
    log_interval = cfg["log_interval"]
    num_steps = cfg['train']['num_steps']
    algo_args = cfg['train']['algorithm_params']
    num_env_steps = cfg['train']['num_env_steps']
    save_interval = cfg["train"]["save_interval"]
    save_path = os.path.join(
        "./checkpoints", task, cfg["algorithm"], cfg["id"], str(seed))

    logger = get_logger(cfg=cfg)
    logger.info(cfg)
    writer = SummaryWriter(log_dir=os.path.join(
        "./tb_logs", task, cfg["algorithm"], cfg["id"]))

    torch.manual_seed(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = get_env(cfg=cfg, num_workers=num_workers, device=device)

    if 'augmentation' not in cfg['train']:
        augmenter = None
        logger.info('No augmenter parameters given. No augmenter set')
    else:
        augmenter = Augmenter(cfg=cfg, device=device)
        logger.info(
            f"Augmenter set. Parameters: {cfg['train']['augmentation']}")

    if cfg['algorithm'] == 'PPO':
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space)
    elif cfg['algorithm'] == 'Aug_PPO':
        actor_critic = AugPolicy(obs_shape=envs.observation_space.shape,
                                 action_space=envs.action_space, augs_list=cfg['augs'])
    else:
        raise NotImplementedError

    actor_critic.to(device)

    if 'PPO' in cfg['algorithm']:
        agent = PPO(actor_critic=actor_critic, **algo_args)
    else:
        raise NotImplementedError(
            f"Given unknown algorithm:{cfg['algorithm']}")

    rollouts = RolloutStorage(num_steps=num_steps, num_processes=num_workers,
                              obs_shape=envs.observation_space.shape, action_space=envs.action_space,
                              recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    max_mean_eval_reward = float('-inf')
    num_updates = int(num_env_steps) // num_steps // num_workers
    logger.info(f"Number of updates is set to: {num_updates}")
    logger.info(f"Training Begins!")

    for j in range(num_updates):
        if not cfg['train']['disable_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer=agent.optimizer, epoch=j, total_num_epochs=num_updates, initial_lr=algo_args['lr'])

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(
            next_value=next_value, **cfg['train']['compute_returns'])

        if 'PPO' in cfg['algorithm']:
            value_loss, action_loss, dist_entropy = agent.update(
                rollouts=rollouts, augmenter=augmenter)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % save_interval == 0 or j == num_updates - 1):
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if j == 0:
                with open(os.path.join(save_path, "config.json"), "w") as output:
                    logger.info(f"num update: {j}. Saving config.yaml...")
                    json.dump(cfg, output)

            logger.info(f"num update: {j}. Saving checkpoint...")

            save_model(save_path=save_path, epoch=j, agent=agent)

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_workers * num_steps
            end = time.time()
            mean_reward = np.mean(episode_rewards).item()
            median_reward = np.median(episode_rewards).item()
            min_reward = np.min(episode_rewards).item()
            max_reward = np.max(episode_rewards).item()
            writer.add_scalar(tag='FPS', scalar_value=int(
                total_num_steps / (end - start)), global_step=total_num_steps)
            writer.add_scalar(tag="Mean Reward Of Last 10 Episode Rewards",
                              scalar_value=mean_reward, global_step=total_num_steps)
            writer.add_scalar(tag="Median Reward Of Last 10 Episode Rewards",
                              scalar_value=median_reward, global_step=total_num_steps)
            writer.add_scalar(tag="Min Reward Of Last 10 Episode Rewards",
                              scalar_value=min_reward, global_step=total_num_steps)
            writer.add_scalar(tag="Max Reward Of Last 10 Episode Rewards",
                              scalar_value=max_reward, global_step=total_num_steps)

            if 'PPO' in cfg['algorithm']:
                writer.add_scalar(tag="Distribution Entropy At Num Step",
                                  scalar_value=dist_entropy, global_step=total_num_steps)
                writer.add_scalar(tag="Value Loss At Num Step",
                                  scalar_value=value_loss, global_step=total_num_steps)
                writer.add_scalar(tag="Action Loss At Num Step",
                                  scalar_value=action_loss, global_step=total_num_steps)

            logger.info(
                f'Step:{total_num_steps}/{int(cfg["train"]["num_env_steps"])}, mean reward: {mean_reward}, median reward: {median_reward}, min reward: {min_reward}, max_reward: {max_reward}')

        if len(episode_rewards) > 1 and j % cfg['eval_interval'] == 0:
            eval_step = int(j // cfg['eval_interval']) + 1
            mean_eval_return = evaluate(actor_critic=actor_critic, cfg=cfg, logger=logger,
                                        num_processes=num_workers, writer=writer, eval_step=eval_step, device=device)

            if mean_eval_return > max_mean_eval_reward:
                max_mean_eval_reward = mean_eval_return
                save_model(save_path=save_path, epoch=j,
                           agent=agent, is_best=True)

    logger.info(
        f'Total Time To Complete: {str(datetime.timedelta(seconds=end - start))}')
    writer.close()


if __name__ == "__main__":
    cfg = get_config()

    if cfg['device_id'] is not None:
        with torch.cuda.device(cfg['device_id']):
            main(cfg)
    else:
        main(cfg)
