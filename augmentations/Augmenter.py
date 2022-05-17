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


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


class Augmenter:
    def __init__(self, cfg: dict, device: str) -> None:
        self.aug_list = cfg['train']['augmentation']['augs']
        self.probs_list = cfg['train']['augmentation']['distribution']
        self.is_crop = False
        self.is_translate = False
        self.crop_sz = None
        self.translate_sz = None
        self.pre_aug_sz = cfg['screen_sz']
        self.batch_sz = cfg['train']['augmentation']['batch_sz']
        self.is_full = cfg['train']['augmentation']['is_full']
        self.device = device

        if len(self.aug_list) != len(self.probs_list):
            raise ValueError(
                'Len of list of augs does not equal number of bins in Categorical distribution')

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
        aug_input = torch.clone(input=input)

        if self.is_crop:
            aug_input = rad.random_crop(
                imgs=aug_input, out=self.crop_sz)

        if self.is_translate:
            centered_input = center_crop_images(
                image=aug_input, output_size=self.pre_aug_width)
            aug_input = rad.random_translate(
                imgs=centered_input, size=self.translate_sz).to(device=self.device)

        if self.num_augs > 0:
            sampled_idxes = np.random.choice(self.num_augs, self.batch_sz)
            unique_values = np.unique(sampled_idxes)

            for value in unique_values:
                idxes_matching = np.where(sampled_idxes == value)[0]
                aug_input[idxes_matching] = self.aug_funcs[value](
                    aug_input[idxes_matching]).to(device=self.device)

        return aug_input
