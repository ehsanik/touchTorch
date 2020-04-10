import torch
import random
import torch.nn.functional as F

from datasets.keypoint_and_trajectory_dataset import DatasetWAugmentation


# For ablation studies, Figure 8 in the paper
class TweakInitialStateDataset(DatasetWAugmentation):
    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):
        super(TweakInitialStateDataset, self).__init__(args, train)
        assert args.mode != 'train'
        self.rotation_tweak = args.rotation_tweak
        self.translation_tweak = args.translation_tweak

    def __getitem__(self, item):
        input, labels = super(TweakInitialStateDataset, self).__getitem__(item)
        rotation_tweaks = torch.rand(4) * torch.Tensor([(-1) ** random.randint(0, 1) for i in range(4)])
        translation_tweaks = torch.rand(3) * torch.Tensor([(-1) ** random.randint(0, 1) for i in range(3)])
        rotation_tweaks = rotation_tweaks / rotation_tweaks.norm() * self.rotation_tweak
        translation_tweaks = translation_tweaks / translation_tweaks.norm() * self.translation_tweak
        input['initial_rotation'] = F.normalize((input['initial_rotation'] + rotation_tweaks).unsqueeze(0)).squeeze(0)
        input['initial_position'] = (input['initial_position'] + translation_tweaks)
        return input, labels
