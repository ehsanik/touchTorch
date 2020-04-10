import argparse
import datetime
import logging
import models
import os
import sys
import torch
import pdb
import pprint
from utils.logging_util import LoggingModule
import utils.loss_util as loss
import time
import environments
import datasets
from utils.constants import CHOSEN_OBJECTS


def loss_class(class_name):
    if class_name not in loss.__all__:
       raise argparse.ArgumentTypeError("Invalid Loss {}; choices: {}".format(
           class_name, loss.__all__))
    return getattr(loss, class_name)

def model_class(class_name):
    if class_name not in models.__all__:
        raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
            class_name, models.__all__))
    return getattr(models, class_name)

def environment_class(class_name):
    if class_name not in environments.__all__:
        raise argparse.ArgumentTypeError("Invalid environment {}; choices: {}".format(
            class_name, environments.__all__))
    return getattr(environments, class_name)

def dataset_class(class_name):
    if class_name not in datasets.__all__:
        raise argparse.ArgumentTypeError("Invalid environment {}; choices: {}".format(
            class_name, datasets.__all__))
    return getattr(datasets, class_name)




def setup_logging(filepath, verbose):
    logFormatter = logging.Formatter(
        '%(levelname)s %(asctime)-20s:\t %(message)s')
    rootLogger = logging.getLogger()
    if verbose:
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Setup the logger to write into file
    fileHandler = logging.FileHandler(filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Setup the logger to write into stdout
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def get_non_default_flags_str(args, parser, *ignore):
    flags = []
    counter = 0
    for key, val in sorted(vars(args).items()):
        if key in ignore:
            continue
        if isinstance(val, type):
            val = val.__name__
        if val != parser.get_default(key):
            flags.append(key + '-' + str(val).replace(' ', '#'))
            counter += 1
        if counter > 5:
            break
    return '+'.join(flags)


def parse_args():

    parser = argparse.ArgumentParser(description='Touch torch')
    parser.add_argument('mode', default='train', nargs='?',choices=('train', 'test', 'testtrain', 'savegtforce'))
    parser.add_argument('--data', metavar='DIR', default='FPHA', help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default='cache', help='path to cache directory')
    parser.add_argument('--environment', default='PhysicsEnv', help='Environment to use for training/test.', type=environment_class)
    parser.add_argument('--dataset', default='EnvironmentDataset', help='Dataset to use for training/test.', type=dataset_class)
    parser.add_argument('--model', '-a', metavar='ARCH', default='SimpleModel', help='model to use for training/test.', type=model_class)
    parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--break-batch', default=64, type=int, help='break batches with this factor to fit to memory.')
    parser.add_argument('--lrm', default=0.1, type=float, help='learning rate multiplier.')
    parser.add_argument('--base-lr', default=0.001, type=float, help='base learning rate ')
    parser.add_argument('--reload', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--reload_dir', default=None, type=str, metavar='PATH')
    parser.add_argument('--reload_from_title', default=None, type=str)
    parser.add_argument('--reload_from_title_epoch', default=-1, type=int)
    parser.add_argument('--no-strict', action='store_false', dest='strict', help='Loading the weights from another model.')
    parser.add_argument('--step_size', default=200, type=int, help='Step size for reducing the learning rate')
    parser.add_argument('--dropout_ratio', default=0.3, type=float)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--number_of_cp', default=5, type=int)
    parser.add_argument('--manual_epoch', default=None, type=int)
    parser.add_argument('--tensorboard_log_freq', default=100, type=int)
    parser.add_argument('--sequence_length', default=10, type=int)
    parser.add_argument('--force_multiplier', default=4, type=float)
    parser.add_argument('--force_h', default=0.01, type=float)
    parser.add_argument('--state_h', default=0.01, type=float)
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--force_clamping', default=1, type=float)
    parser.add_argument('--predicted_cp_adr', default=None, type=str)
    parser.add_argument('--subsample_rate', default=1, type=int)
    parser.add_argument('--rotation_tweak', default=None, type=float)
    parser.add_argument('--translation_tweak', default=None, type=float)
    parser.add_argument('--save_frequency', default=1, type=int, help='Frequency of saving the model, per epoch')
    parser.add_argument('--title', default='')
    parser.add_argument('--loss', default='ObjectTrajectoryLoss', type=loss_class)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no_gravity', action='store_false', dest='gravity')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_val', action='store_true')
    parser.add_argument('--qualitative_size', default=500, type=int)
    parser.add_argument('--no-pretrain', action='store_false', dest='pretrain')
    parser.add_argument('--object_list', default=['072-a_toy_airplane'], nargs='+', type=str, help='options: ALL or ycb objects')
    parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')


    args = parser.parse_args()

    args.logdir = args.data

    args.save = os.path.join(args.logdir, args.save)

    if args.gpu_ids == [-1]:
        args.gpu_ids = -1

    assert sum([args.reload is not None, args.reload_dir is not None, args.reload_from_title is not None]) <= 1

    if args.gpu_ids != -1:
        torch.cuda.manual_seed(args.seed)

    logging_path = os.path.join(args.logdir, 'runs/')
    local_start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    log_title = args.title + '_' + local_start_time_str
    log_dir = os.path.join(logging_path,  log_title)
    args.qualitative_dir = os.path.join(args.logdir, 'qualitative_plots', log_title)

    timestamp = str(datetime.datetime.now()).replace(' ', '#').replace(':', '.')
    args.timestamp = timestamp
    args.save = os.path.join(
        args.save, args.model.__name__, log_title, 
        get_non_default_flags_str(args, parser, 'data', 'save', 'model',
                                  'reload', 'title', 'workers', 'save_frequency', 'batch-size', 'gpu-ids'), timestamp)
    os.makedirs(args.save, exist_ok=True)
    setup_logging(os.path.join(args.save, 'log.txt'), True)

    if args.object_list in [['ALL']]:
        if args.object_list == ['ALL']:
            args.object_list = CHOSEN_OBJECTS
        else:
            raise Exception ('Not implemented yet')

    logging.info('Command: {}'.format(' '.join(sys.argv)))
    logging.info('Command line arguments parsed: {}'.format(
        pprint.pformat(vars(args))))

    args.logging_module = LoggingModule(args, log_dir)

    return args
