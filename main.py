import torch
import logging
import random
import os
import matplotlib as mpl

mpl.use('Agg')
from pathlib import Path
import numpy as np

from solvers import train, test, save_gt_force

from utils.arg_parser import parse_args


def get_dataset(args):
    train_dataset = args.dataset(args, train=True)
    val_dataset = args.dataset(args, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)
    test_shuffle = True
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=test_shuffle, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def get_model_and_loss(args):
    model = args.model(args)
    restarting_epoch = 0
    if args.gpu_ids != -1:
        model = model.cuda()
    reload_adr = None
    args.final_reload = None
    if args.reload:
        reload_adr = args.reload
    elif args.reload_from_title is not None:
        file = [f for f in Path(os.path.join(args.data, 'cache')).glob('**/' + args.reload_from_title)]
        assert len(file) == 1
        file = file[0]
        all_saved_models = [str(f) for f in file.glob('**/*.pytar')]
        epoch_indices = [int(mod.split('_')[-1].replace('.pytar', '')) for mod in all_saved_models]
        if args.reload_from_title_epoch > 0:
            latest_index = epoch_indices.index(args.reload_from_title_epoch)
        else:
            latest_index = np.argmax(np.array(epoch_indices))
        reload_adr = all_saved_models[latest_index]
        print('Exact Address is:', reload_adr)
    if reload_adr is not None:
        if args.gpu_ids == -1:
            loaded_weights = torch.load(reload_adr, map_location='cpu')
        else:
            loaded_weights = torch.load(reload_adr)
        args.final_reload = reload_adr
        model.load_state_dict(loaded_weights, strict=args.strict)
        epoch_index = reload_adr.split('_')[-1].replace('.pytar', '')
        try:
            epoch_index = int(epoch_index)
        except Exception:
            epoch_index = 0
        restarting_epoch = epoch_index
        print('Restarting from epoch', restarting_epoch)
    if not args.strict:
        restarting_epoch = 0

    if args.manual_epoch is not None:
        restarting_epoch = args.manual_epoch
        print('Manually setting the epoch', restarting_epoch)

    loss = model.loss(args)
    if args.gpu_ids != -1:
        loss = loss.cuda()
    logging.info('Model: {}'.format(model))
    logging.info('Loss: {}'.format(loss))
    return model, loss, restarting_epoch


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Reading dataset metadata')
    train_loader, val_loader = get_dataset(args)

    logging.info('Constructing model')
    model, loss, restarting_epoch = get_model_and_loss(args)

    if args.mode == 'train':
        optimizer = model.optimizer()
        for i in range(restarting_epoch, args.epochs):
            print('Epoch[', i, ']')
            train.train_one_epoch(model, loss, optimizer, train_loader, i + 1,
                                  args)
            if i % args.save_frequency == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save,
                                 'model_state_{:02d}.pytar'.format(i + 1)))
            test.test_one_epoch(model, loss, val_loader, i + 1, args)
    elif args.mode == 'test' or args.mode == 'testtrain':
        if args.mode == 'testtrain':
            val_loader = train_loader
        if args.reload_dir is not None:
            all_saved_models = [f for f in os.listdir(args.reload_dir) if f.endswith('.pytar')]
            all_indices = [f.split('_')[-1].replace('.pytar', '') for f in all_saved_models]
            int_indices = [int(f) for f in all_indices]
            int_indices.sort()
            for epoch in int_indices:
                args.reload = os.path.join(args.reload_dir, 'model_state_{:02d}.pytar'.format(epoch))
                print('Loaded ', args.reload, 'epoch', epoch)
                model, loss, restarting_epoch = get_model_and_loss(args)
                test.test_one_epoch(model, loss, val_loader, epoch, args)
        else:
            test.test_one_epoch(model, loss, val_loader, 0, args)
    elif args.mode == 'savegtforce':
        save_gt_force.save_gt_force(model, loss, train_loader, 0, args)
    else:
        raise NotImplementedError("Unsupported mode {}".format(args.mode))


if __name__ == "__main__":
    main()
