from tensorboardX import SummaryWriter
import torch
import os
import random
import matplotlib as mpl
from utils.visualization_util import normalize

mpl.use('Agg')

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO


class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means


class LoggingModule(object):
    def __init__(self, args, log_dir):
        print('initializing logger', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.qualitative_dir = args.qualitative_dir
        self.mode = args.mode
        self.log_writer = SummaryWriter(log_dir=log_dir)
        self.number_of_items_to_visualize = 10
        self.render = args.render

    def recursive_write(self, item_to_write, epoch_number, add_to_keys=''):
        if type(item_to_write) == dict:
            for res in item_to_write:
                sub_item = item_to_write[res]
                new_translated_key = add_to_keys + '/' + res
                self.recursive_write(sub_item, epoch_number, add_to_keys=new_translated_key)
        elif type(item_to_write) == torch.Tensor and len(item_to_write.shape) == 2:
            seq_len, joint_len = item_to_write.shape
            for seq_ind in range(seq_len):
                sub_item = item_to_write[seq_ind].mean()
                new_translated_key = add_to_keys + '/' + 'seq_ind_{}'.format(seq_ind)
                self.recursive_write(sub_item, epoch_number, add_to_keys=new_translated_key)
            for joint_ind in range(joint_len):
                sub_item = item_to_write[:, joint_ind].mean()
                new_translated_key = add_to_keys + '/' + 'joint_ind_{}'.format(joint_ind)
                self.recursive_write(sub_item, epoch_number, add_to_keys=new_translated_key)
            self.recursive_write(item_to_write.mean(), epoch_number, add_to_keys=add_to_keys + '/' + 'overall')
        elif type(item_to_write) == torch.Tensor and len(item_to_write.shape) == 1:
            seq_len = len(item_to_write)  # equal to item_to_write.shape[0]
            for seq_ind in range(seq_len):
                sub_item = item_to_write[seq_ind]
                new_translated_key = add_to_keys + '/' + 'seq_ind_{}'.format(seq_ind)
                self.recursive_write(sub_item, epoch_number, add_to_keys=new_translated_key)
            self.recursive_write(item_to_write.mean(), epoch_number, add_to_keys=add_to_keys + '/' + 'overall')
        else:
            self.log_writer.add_scalar(
                add_to_keys, item_to_write, epoch_number
            )

    def subplot_summary(self, subplot_image_sequence, step, add_to_keys):
        sequence_length = len(subplot_image_sequence)
        for seq_index in range(sequence_length):
            output = subplot_image_sequence[seq_index].transpose(2, 0, 1)
            self.log_writer.add_image(tag="%s/time_%d_output" % (add_to_keys, step), img_tensor=output, global_step=step + seq_index)

    def image_summary(self, output_images, target_images, step, add_to_keys):

        batch_size, sequence_length, _, _, _ = output_images.shape

        # img_summaries = []
        batch_to_visualize = random.randint(0, batch_size - 1)
        for seq_index in range(sequence_length):
            output = output_images[batch_to_visualize, seq_index].cpu()
            target = target_images[batch_to_visualize, seq_index].cpu()

            output = normalize(output)
            target = normalize(target)

            self.log_writer.add_image(tag="%s/time_%d_output" % (add_to_keys, seq_index), img_tensor=output, global_step=step)
            self.log_writer.add_image(tag="%s/time_%d_target" % (add_to_keys, seq_index), img_tensor=target, global_step=step)

    def log(self, dict_res, epoch_number, add_to_keys=''):
        for k in dict_res:
            if add_to_keys != '':
                translated_k = k + '/' + add_to_keys
            else:
                translated_k = k
            self.recursive_write(dict_res[k], epoch_number, add_to_keys=translated_k)
