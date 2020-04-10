import logging
import os
import time
import torch
import pdb
from . import metrics
import tqdm


def test_one_epoch(model, loss, data_loader, epoch, args):
    add_to_keys = 'Test'

    # Prepare model and optimizer
    model.eval()
    loss.eval()
    with torch.no_grad():
        # Setup average meters
        data_time_meter = metrics.AverageMeter()
        batch_time_meter = metrics.AverageMeter()
        metrics_time_meter = metrics.AverageMeter()
        backward_time_meter = metrics.AverageMeter()
        forward_pass_time_meter = metrics.AverageMeter()
        loss_time_meter = metrics.AverageMeter()
        loss_meter = metrics.AverageMeter()
        accuracy_metric = [m(args) for m in model.metric]
        loss_detail_meter = {loss_name: metrics.AverageMeter() for loss_name in loss.local_loss_dict}

        # Iterate over data
        timestamp = time.time()
        for i, (input, target) in enumerate(tqdm.tqdm(data_loader)):

            # Move data to gpu
            batch_size = input['rgb'].size(0)
            if args.gpu_ids != -1:
                for feature in input:
                    value = input[feature]
                    if issubclass(type(value), torch.Tensor):
                        input[feature] = value.cuda(async=True)
                target = {feature: target[feature].cuda(async=True) for feature in target.keys()}
            data_time_meter.update((time.time() - timestamp) / batch_size, batch_size)

            before_forward_pass_time = time.time()
            # Forward pass
            output, target_output = model(input, target)
            forward_pass_time_meter.update((time.time() - before_forward_pass_time) / batch_size, batch_size)

            before_loss_time = time.time()
            loss_output = loss(output, target_output)
            loss_time_meter.update((time.time() - before_loss_time) / batch_size, batch_size)
            if args.render:
                print('loss', loss.local_loss_dict)

            before_backward_time = time.time()
            backward_time_meter.update((time.time() - before_backward_time) / batch_size, batch_size)

            output = {f: output[f].detach() for f in output.keys()}

            # Bookkeeping on loss, accuracy, and batch time
            loss_meter.update(loss_output.detach(), batch_size)
            before_metrics_time = time.time()
            with torch.no_grad():
                for acc in accuracy_metric:
                    acc.record_output(output, target_output)
            metrics_time_meter.update((time.time() - before_metrics_time) / batch_size, batch_size)
            batch_time_meter.update((time.time() - timestamp), batch_size)

            # Log report
            dataset_length = len(data_loader.dataset)
            real_index = (epoch - 1) * dataset_length + (i * args.batch_size)

            loss_values = loss.local_loss_dict
            for loss_name in loss_detail_meter:
                (loss_val, data_size) = loss_values[loss_name]
                loss_detail_meter[loss_name].update(loss_val.item(), data_size)

            if i % args.tensorboard_log_freq == 0:
                result_log_dict = {
                    'Time/Batch': batch_time_meter.avg,
                    'Time/Data': data_time_meter.avg,
                    'Time/Metrics': metrics_time_meter.avg,
                    'Time/backward': backward_time_meter.avg,
                    'Time/forward': forward_pass_time_meter.avg,
                    'Time/loss': loss_time_meter.avg,
                    'Loss': loss_meter.avg,
                }

                for loss_name in loss_detail_meter:
                    result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()
                args.logging_module.log(result_log_dict, real_index + 1, add_to_keys=add_to_keys)

            timestamp = time.time()

        result_log_dict = {
            'Time/Batch': batch_time_meter.avg,
            'Time/Data': data_time_meter.avg,
            'Time/Metrics': metrics_time_meter.avg,
            'Time/backward': backward_time_meter.avg,
            'Time/forward': forward_pass_time_meter.avg,
            'Time/loss': loss_time_meter.avg,
            'Loss': loss_meter.avg,
        }

        for loss_name in loss_detail_meter:
            result_log_dict['Loss/' + loss_name] = loss_detail_meter[loss_name].avg

        with torch.no_grad():
            for ac in accuracy_metric:
                result_log_dict[type(ac).__name__] = ac.average()
        args.logging_module.log(result_log_dict, epoch, add_to_keys=add_to_keys + '_Summary')

        testing_summary = ('Epoch: [{}] -- TESTING SUMMARY\t'.format(epoch) + 'Time {batch_time.sum:.2f}   Data {data_time.sum:.2f}  Loss {loss.avg:.6f}   {accuracy_report}'.format(batch_time=batch_time_meter, data_time=data_time_meter, loss=loss_meter, accuracy_report='\n'.join([ac.final_report() for ac in accuracy_metric])))

    logging.info(testing_summary)
    logging.info('Full test result is at {}'.format(
        os.path.join(args.save, 'test.log')))

    with open(os.path.join(args.save, 'test.log'), 'a') as fp:
        fp.write('{}\n'.format(testing_summary))
