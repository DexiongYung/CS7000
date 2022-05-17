import numpy as np
import torch
from env import get_env
from augmentations.Augmenter import create_aug_func_dict


def evaluate(actor_critic, cfg, num_processes, device, eval_step, writer=None, logger=None):
    augs_dict = create_aug_func_dict(
        augs_list=cfg['validation']['augmentation'])
    mean_reward_list = list()

    for aug_key, aug_func in augs_dict.items():
        eval_envs = get_env(cfg=cfg, num_workers=num_processes,
                            device=device, seed=cfg['seed'] + 1000, is_eval=True)
        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)

        while len(eval_episode_rewards) < 10:
            aug_obs = aug_func(obs)
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    aug_obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        eval_envs.close()

        mean_reward = np.mean(eval_episode_rewards).item()
        median_reward = np.median(eval_episode_rewards).item()
        min_reward = np.min(eval_episode_rewards).item()
        max_reward = np.max(eval_episode_rewards).item()

        if writer is not None:
            writer.add_scalar(tag=f"Eval {aug_key} Mean Reward Of Last 10 Episode Rewards",
                              scalar_value=mean_reward, global_step=eval_step)
            writer.add_scalar(tag=f"Eval {aug_key} Median Reward Of Last 10 Episode Rewards",
                              scalar_value=median_reward, global_step=eval_step)
            writer.add_scalar(tag=f"Eval {aug_key} Min Reward Of Last 10 Episode Rewards",
                              scalar_value=min_reward, global_step=eval_step)
            writer.add_scalar(tag=f"Eval {aug_key} Max Reward Of Last 10 Episode Rewards",
                              scalar_value=max_reward, global_step=eval_step)

        if logger is not None:
            logger.info(
                f'Eval {aug_key} Number:{eval_step}, mean reward: {mean_reward}, median reward: {median_reward}, min reward: {min_reward}, max_reward: {max_reward}')

        mean_reward_list.append(mean_reward)

    return np.mean(mean_reward_list)
