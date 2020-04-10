import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Function
from utils.environment_util import EnvState, ForceValOnly
from torch.autograd import Variable
import random


def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
        nn.LeakyReLU()
    )


def upshufflenorelu(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    )


def imu_un_embed(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, 64),
        nn.LeakyReLU(),
        nn.Linear(64, out_planes),
    )


def combine_block(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.GroupNorm(32, out_planes),
        nn.LeakyReLU(),
    )


def combine_block_w_do(in_planes, out_planes, dropout=0.):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


def linear_block(in_features, out_features, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        # nn.GroupNorm(group_norm_size, out_features),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


def linear_block_norelu(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        # nn.GroupNorm(group_norm_size, out_features),
        # nn.Dropout(dropout),
    )


def input_embedding_net(list_of_feature_sizes, dropout=0.0):
    modules = []
    for i in range(len(list_of_feature_sizes) - 1):
        input_size, output_size = list_of_feature_sizes[i:i + 2]
        if i + 2 == len(list_of_feature_sizes):
            modules.append(linear_block_norelu(input_size, output_size))
        else:
            modules.append(linear_block(input_size, output_size, dropout=dropout))
    return nn.Sequential(*modules)


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


def replace_all_relu_w_leakyrelu(model):
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        if isinstance(module, nn.ReLU):
            model._modules[m] = nn.LeakyReLU()
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_relu_w_leakyrelu(module)
    return model


def replace_all_bn_w_groupnorm(model):
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            feature_number = module.num_features
            model._modules[m] = nn.GroupNorm(32, feature_number)
        elif isinstance(module, nn.BatchNorm3d):
            raise Exception('Not implemented')
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_bn_w_groupnorm(module)
    return model


def flat_temporal(tensor, batch_size, sequence_length):
    tensor_shape = [s for s in tensor.shape]
    assert tensor_shape[0] == batch_size and tensor_shape[1] == sequence_length
    result_shape = [batch_size * sequence_length] + tensor_shape[2:]
    return tensor.contiguous().view(result_shape)


def unflat_temporal(tensor, batch_size, sequence_length):
    tensor_shape = [s for s in tensor.shape]
    assert tensor_shape[0] == batch_size * sequence_length
    result_shape = [batch_size, sequence_length] + tensor_shape[1:]
    return tensor.contiguous().view(result_shape)


class EnvWHumanCpFiniteDiffFast(Function):
    @staticmethod
    def forward(ctx, environment, initial_state, forces_tensor, list_of_contact_points):
        assert len(forces_tensor.shape) == 1, 'The force should be flattened'
        initial_state = EnvState.fromTensor(initial_state)

        forces = ForceValOnly.fromForceArray(forces_tensor.view(environment.number_of_cp, 3))

        finite_diff_success_flags = []

        with torch.no_grad():
            env_state, force_success_flag, force_that_applied = environment.init_location_and_apply_force(initial_state=initial_state, forces=forces, list_of_contact_points=list_of_contact_points)
            f_x_env_state_tensor = env_state.toTensor()

        no_grad_on = not (True in ctx.needs_input_grad)

        if not no_grad_on:  # at least one of them needs gradients
            with torch.no_grad():

                force_h = environment.force_h
                manual_tweaks_force = []
                for arg_ind in range(len(forces_tensor)):

                    changed_force = forces_tensor * 1  # Just copy
                    tweak_value_force = force_h
                    if random.random() > 0.5:
                        tweak_value_force *= -1

                    changed_force[arg_ind] += tweak_value_force

                    changed_force = ForceValOnly.fromForceArray(changed_force.view(environment.number_of_cp, -1))

                    env_state_h, force_success_flag_h, force_that_applied_h = environment.init_location_and_apply_force(initial_state=initial_state, forces=changed_force, list_of_contact_points=list_of_contact_points)

                    manual_tweaks_force.append((env_state_h.toTensor() - f_x_env_state_tensor) / (tweak_value_force))

                    finite_diff_success_flags.append(force_success_flag_h)

                ctx.f_x_plus_h_diff = torch.stack(manual_tweaks_force, dim=-2)

                state = initial_state.toTensor()
                manual_tweaks_initial_state = []
                state_h = environment.state_h
                for arg_ind in range(len(state)):
                    if arg_ind == EnvState.OBJECT_TYPE_INDEX:
                        manual_tweaks_initial_state.append(torch.zeros([len(state)]))
                        continue

                    changed_state = state * 1  # just to copy

                    tweak_value_state = state_h
                    if random.random() >= 0.5:
                        tweak_value_state *= -1

                    changed_state[arg_ind] += tweak_value_state

                    env_state_h, force_success_flag_h, force_that_applied_h = environment.init_location_and_apply_force(initial_state=EnvState.fromTensor(changed_state), forces=forces, list_of_contact_points=list_of_contact_points)

                    manual_tweaks_initial_state.append((env_state_h.toTensor() - f_x_env_state_tensor) / (tweak_value_state))

                    finite_diff_success_flags.append(force_success_flag_h)
                ctx.initial_state_plus_h_diff = torch.stack(manual_tweaks_initial_state, dim=-2)

        finite_diff_success_flags += [force_success_flag]
        finite_diff_success_flags = torch.Tensor(finite_diff_success_flags)
        force_that_applied = torch.stack(force_that_applied, dim=0)

        env_state_tensor = env_state.toTensor()
        device = forces_tensor.device
        env_state_tensor = env_state_tensor.to(device=device)
        if not no_grad_on:
            ctx.f_x_plus_h_diff = ctx.f_x_plus_h_diff.to(device=device)
            ctx.initial_state_plus_h_diff = ctx.initial_state_plus_h_diff.to(device=device)

        return env_state_tensor, finite_diff_success_flags, force_that_applied

    @staticmethod
    def backward(ctx, env_state_grad, unwanted_finite_diff_success_flags, unwanted_contact_points):
        grad_output = env_state_grad.unsqueeze(1)
        force_gradient = torch.mm(ctx.f_x_plus_h_diff, grad_output).squeeze(1)
        initial_state_gradient = torch.mm(ctx.initial_state_plus_h_diff, grad_output).squeeze(1)

        return (
            None,  # environment
            initial_state_gradient,  # initial_state
            force_gradient,  # force
            None,  # list_of_contact_points
        )


class CPGradientLayer(Function):
    @staticmethod
    def forward(ctx, environment, initial_state, forces_tensor, list_of_contact_points):
        assert len(forces_tensor.shape) == 1, 'The force should be flattened'
        initial_state = EnvState.fromTensor(initial_state)
        forces = ForceValOnly.fromForceArray(forces_tensor.view(environment.number_of_cp, 3))
        list_of_contact_points = list_of_contact_points.view(5, 3)

        finite_diff_success_flags = []

        with torch.no_grad():
            env_state, force_success_flag, force_that_applied = environment.init_location_and_apply_force(initial_state=initial_state, forces=forces, list_of_contact_points=list_of_contact_points)
            f_x_env_state_tensor = env_state.toTensor()

        no_grad_on = not (True in ctx.needs_input_grad)

        if not no_grad_on:  # at least one of them needs gradients
            with torch.no_grad():

                cp_h = 0.05
                # position and linear velocity should not depend, force similar, just angular
                manual_tweaks_cp = []
                contact_point_tensor = list_of_contact_points.view(-1)
                for arg_ind in range(len(contact_point_tensor)):

                    changed_contact_point = contact_point_tensor * 1  # Just copy
                    tweak_value_cp = cp_h
                    if random.random() > 0.5:
                        tweak_value_cp *= -1

                    changed_contact_point[arg_ind] += tweak_value_cp

                    changed_contact_point = changed_contact_point.view(5, 3)

                    env_state_h, _, _ = environment.init_location_and_apply_force(initial_state=initial_state, forces=forces, list_of_contact_points=changed_contact_point)

                    manual_tweaks_cp.append((env_state_h.toTensor() - f_x_env_state_tensor) / (tweak_value_cp))

                ctx.contact_pt_x_plus_h_diff = torch.stack(manual_tweaks_cp, dim=-2)

                force_h = environment.force_h
                manual_tweaks_force = []
                for arg_ind in range(len(forces_tensor)):

                    changed_force = forces_tensor * 1  # Just copy
                    tweak_value_force = force_h
                    if random.random() > 0.5:
                        tweak_value_force *= -1

                    changed_force[arg_ind] += tweak_value_force

                    changed_force = ForceValOnly.fromForceArray(changed_force.view(environment.number_of_cp, -1))

                    env_state_h, force_success_flag_h, force_that_applied_h = environment.init_location_and_apply_force(initial_state=initial_state, forces=changed_force, list_of_contact_points=list_of_contact_points)

                    manual_tweaks_force.append((env_state_h.toTensor() - f_x_env_state_tensor) / (tweak_value_force))

                    finite_diff_success_flags.append(force_success_flag_h)

                ctx.f_x_plus_h_diff = torch.stack(manual_tweaks_force, dim=-2)

                state = initial_state.toTensor()
                manual_tweaks_initial_state = []
                state_h = environment.state_h
                for arg_ind in range(len(state)):
                    if arg_ind == EnvState.OBJECT_TYPE_INDEX:
                        manual_tweaks_initial_state.append(torch.zeros([len(state)]))
                        continue

                    changed_state = state * 1  # just to copy

                    tweak_value_state = state_h
                    if random.random() >= 0.5:
                        tweak_value_state *= -1

                    changed_state[arg_ind] += tweak_value_state

                    env_state_h, force_success_flag_h, force_that_applied_h = environment.init_location_and_apply_force(initial_state=EnvState.fromTensor(changed_state), forces=forces, list_of_contact_points=list_of_contact_points)

                    manual_tweaks_initial_state.append((env_state_h.toTensor() - f_x_env_state_tensor) / (tweak_value_state))

                    finite_diff_success_flags.append(force_success_flag_h)
                ctx.initial_state_plus_h_diff = torch.stack(manual_tweaks_initial_state, dim=-2)

        finite_diff_success_flags += [force_success_flag]
        finite_diff_success_flags = torch.Tensor(finite_diff_success_flags)
        force_that_applied = torch.stack(force_that_applied, dim=0)

        env_state_tensor = env_state.toTensor()
        device = forces_tensor.device
        env_state_tensor = env_state_tensor.to(device=device)
        if not no_grad_on:
            ctx.f_x_plus_h_diff = ctx.f_x_plus_h_diff.to(device=device)
            ctx.initial_state_plus_h_diff = ctx.initial_state_plus_h_diff.to(device=device)
            ctx.contact_pt_x_plus_h_diff = ctx.contact_pt_x_plus_h_diff.to(device=device)

        return env_state_tensor, finite_diff_success_flags, force_that_applied

    @staticmethod
    def backward(ctx, env_state_grad, unwanted_finite_diff_success_flags, unwanted_contact_points):
        grad_output = env_state_grad.unsqueeze(1)
        # Do NOT use unwanted_contact_points, it does not make any sense
        force_gradient = torch.mm(ctx.f_x_plus_h_diff, grad_output).squeeze(1)
        initial_state_gradient = torch.mm(ctx.initial_state_plus_h_diff, grad_output).squeeze(1)
        contact_point_gradient = torch.mm(ctx.contact_pt_x_plus_h_diff, grad_output).squeeze(1)
        return (
            None,  # environment
            initial_state_gradient,  # initial_state
            force_gradient,  # force
            contact_point_gradient,  # list_of_contact_points
        )
