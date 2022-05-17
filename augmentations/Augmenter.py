import torch
import numpy as np
import augmentations.rad as rad
import augmentations.new_augs as new_augs

aug_to_func = {
    'grayscale': rad.random_grayscale,
    'cutout': rad.random_cutout,
    'cutout_color': rad.random_cutout_color,
    'flip': rad.random_flip,
    'rotate': rad.random_rotation,
    'rand_conv': rad.random_convolution,
    # 'color_jitter': rad.random_color_jitter,
    'no_aug': rad.no_aug,
    'blackout': new_augs.blackout
}


def create_aug_func_dict(augs_list: list):
    augs_func_dict = dict()
    for aug_name in augs_list:
        assert aug_name in aug_to_func.keys()
        augs_func_dict[aug_name] = aug_to_func[aug_name]

    return augs_func_dict


def create_aug_func_list(augs_list: list):
    augs_func_list = list()
    for i in range(len(augs_list)):
        aug_name = augs_list[i]
        assert aug_name in aug_to_func.keys()
        augs_func_list.append(aug_to_func[aug_name])

    return augs_func_list


class Augmenter:
    def __init__(self, cfg: dict, device: str) -> None:
        # TODO: Add post aug sizing to CFG
        self.aug_list = cfg['train']['augmentation']['augs']
        self.probs_list = cfg['train']['augmentation']['distribution']
        self.is_crop = False
        self.is_translate = False
        self.crop_sz = None
        self.translate_sz = None
        self.pre_aug_width = cfg['screen_width']
        self.pre_aug_height = cfg['screen_height']

        if len(self.aug_list) != len(self.probs_list):
            raise ValueError(
                'Len of list of augs does not equal number of bins in Categorical distribution')

        self.batch_sz = cfg['train']['augmentation']['batch_sz']
        self.is_full = cfg['train']['augmentation']['is_full']
        self.device = device

        self.aug_funcs = list()

        if 'crop' in cfg['train']['augmentation'] and cfg['train']['augmentation']['crop']:
            self.is_crop = True
            self.crop_sz = cfg['train']['augmentation']['crop_sz']
            print('Crop is on!')

        if 'translate' in cfg['train']['augmentation'] and cfg['train']['augmentation']['translate']:
            self.is_translate = True
            self.translate_sz = cfg['train']['augmentation']['translate_sz']
            print('Translate is on!')

        if cfg['algorithm'] != 'Aug_PPO':
            self.aug_funcs = create_aug_func_list(augs_list=self.aug_list)

        self.num_augs = len(self.aug_funcs)
        print(f'Aug set is: {self.aug_list}, with size: {self.num_augs}')

    def augment_tensors_in_batches(self, input):
        if self.is_full:
            self.batch_sz = input.shape[0]
        else:
            sampled_batch_idxes = np.random.choice(
                input.shape[0], self.batch_sz)
            input = input[sampled_batch_idxes]

        sampled_idxes = np.random.choice(self.num_augs, self.batch_sz)
        unique_values = np.unique(sampled_idxes)
        aug_input = torch.zeros(input.shape).to(self.device)

        if self.is_crop:
            aug_input = rad.random_crop(imgs_tnsr=aug_input, out=self.crop_sz)

        if self.is_translate:
            aug_input = rad.random_translate(
                imgs=aug_input, size=self.translate_sz)

        for value in unique_values:
            idxes_matching = np.where(sampled_idxes == value)[0]
            aug_input[idxes_matching] = self.aug_funcs[value](
                input[idxes_matching]).to(self.device)

        return aug_input
