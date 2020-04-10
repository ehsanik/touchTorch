"""
=================
This is the basic model containing place holder/implementation for necessary functions.

All the models in this project inherit from this class
=================
"""

import torch.nn as nn
import logging


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.base_lr = args.base_lr
        self.lrm = args.lrm
        self.step_size = args.step_size

    def loss(self, args):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        # Learning rate here is just a place holder. This will be overwritten
        # at training time.
        raise NotImplementedError

    def learning_rate(self, epoch):
        base_lr = self.base_lr
        decay_rate = self.lrm
        step = self.step_size
        assert 1 <= epoch
        if 1 <= epoch <= step:
            return base_lr
        elif step <= epoch <= step * 2:
            return base_lr * decay_rate
        elif step * 2 <= epoch <= step * 3:
            return base_lr * decay_rate * decay_rate
        else:
            return base_lr * decay_rate * decay_rate * decay_rate

    def evaluation_report(self, output, target):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.
        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """

        own_state = self.state_dict()
        copied = []
        for name, param in state_dict.items():
            original_name = name
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    print('Copied {} to {}'.format(original_name, name))
                    own_state[name].copy_(param)
                    copied.append(name)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            else:
                logging.warning(
                    'Parameter {} not found in own state'.format(original_name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))
        else:
            missing = set(own_state.keys()) - set(copied)
            # missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                logging.warning(
                    'missing keys in state_dict: "{}"'.format(missing))
