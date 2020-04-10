import os
import torch
import json
import tqdm


def save_gt_force(model, loss, data_loader, epoch, args):
    time_to_force_dict = {}

    assert args.batch_size == 1

    # Prepare model and optimizer
    model.eval()
    loss.eval()
    accuracy_metric = [m(args) for m in model.metric]

    if True:
        # Setup average meters

        for i, (input, target) in enumerate(tqdm.tqdm(data_loader)):

            # Move data to gpu
            if args.gpu_ids != -1:
                for feature in input:
                    value = input[feature]
                    if issubclass(type(value), torch.Tensor):
                        input[feature] = value.cuda(async=True)
                target = {feature: target[feature].cuda(async=True) for feature in target.keys()}

            # Forward pass
            output, target_output = model(input, target)

            if i % args.tensorboard_log_freq == 0:
                result_log_dict = {
                }

                with torch.no_grad():
                    for acc in accuracy_metric:
                        acc.record_output(output, target_output)

                for ac in accuracy_metric:
                    result_log_dict[type(ac).__name__] = ac.average()
                args.logging_module.log(result_log_dict, i + 1, add_to_keys='Test')

            sequence_str = '__'.join([time[0] for time in input['timestamps']])
            force = output['force_direction'][0]
            time_to_force_dict[sequence_str] = force.cpu().detach().tolist()

    adr_to_save_cp = os.path.join(args.save, 'gtforce_train.json')
    print('Saving contact points to ', adr_to_save_cp)
    with open(adr_to_save_cp, 'w') as file:
        json.dump(time_to_force_dict, file)
