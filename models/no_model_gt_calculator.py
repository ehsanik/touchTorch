import torch
import torch.nn as nn
from .base_model import BaseModel
from utils.net_util import EnvWHumanCpFiniteDiffFast

from solvers import metrics
from utils.environment_util import EnvState
from utils.projection_utils import get_keypoint_projection, get_all_objects_keypoint_tensors


class NoModelGTForceBaseline(BaseModel):
    metric = [
        metrics.ObjKeypointMetric,
    ]

    def __init__(self, args):
        super(NoModelGTForceBaseline, self).__init__(args)
        self.environment_layer = EnvWHumanCpFiniteDiffFast
        self.loss_function = args.loss
        self.number_of_cp = args.number_of_cp
        self.environment = args.instance_environment
        self.sequence_length = args.sequence_length
        self.gpu_ids = args.gpu_ids

        self.dummy_layer = nn.Linear(10, 10)

        self.this_loss_func = self.loss(args)
        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

    def loss(self, args):
        return self.loss_function(args)

    def forward(self, input, target):

        initial_position = input['initial_position']
        initial_rotation = input['initial_rotation']
        contact_points = input['contact_points']
        assert contact_points.shape[0] == 1
        contact_points = contact_points.squeeze(0)

        object_name = input['object_name']
        assert len(object_name) == 1
        object_name = object_name[0]
        target['object_name'] = input['object_name']

        predefined_force = target['forces'].squeeze(0).view(self.sequence_length - 1, 5 * 3)
        all_forces = torch.nn.Parameter(predefined_force.detach())

        loss_function = self.this_loss_func

        if self.gpu_ids != -1:
            all_forces = all_forces.cuda().detach()
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD([all_forces], lr=self.base_lr)

        number_of_states = 20

        for t in range(number_of_states):

            # all_forces = all_forces.clamp(-1.5, 1.5)

            if t <= self.step_size:
                lr = self.base_lr
            elif t <= self.step_size * 2:
                lr = self.base_lr * 0.1
            elif t <= self.step_size * 3:
                lr = self.base_lr * 0.01

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            env_state = EnvState(object_name=object_name, rotation=initial_rotation[0], position=initial_position[0], velocity=None, omega=None)
            resulting_force_success = []
            forces_directions = []
            all_force_applied = []
            resulting_position = []
            resulting_rotation = []

            for seq_ind in range(self.sequence_length - 1):
                force = all_forces[seq_ind]

                assert force.shape[0] == (self.number_of_cp * 3)

                # initial initial_velocity is whatever it was the last frame, note that the gradients are not backproped here
                env_state, force_success, force_applied = self.environment_layer.apply(self.environment, env_state.toTensor(), force, contact_points)
                env_state = EnvState.fromTensor(env_state)

                resulting_position.append(env_state.position)
                resulting_rotation.append(env_state.rotation)
                resulting_force_success.append(force_success)
                forces_directions.append(force.view(self.number_of_cp, 3))
                all_force_applied.append(force_applied)

            resulting_position = torch.stack(resulting_position, dim=0)
            resulting_rotation = torch.stack(resulting_rotation, dim=0)

            resulting_position = resulting_position.unsqueeze(0)  # adding batchsize back because we need it in the loss
            resulting_rotation = resulting_rotation.unsqueeze(0)  # adding batchsize back because we need it in the loss

            all_keypoints = get_keypoint_projection(object_name, resulting_position, resulting_rotation, self.all_objects_keypoint_tensor[object_name])
            all_keypoints = all_keypoints.unsqueeze(0)  # adding batchsize back because we need it in the loss

            output = {
                'keypoints': all_keypoints,
            }

            loss = loss_function(output, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        resulting_force_success = torch.stack(resulting_force_success, dim=0)
        forces_directions = torch.stack(forces_directions, dim=0)
        all_force_applied = torch.stack(all_force_applied, dim=0)

        forces_directions = forces_directions.unsqueeze(0)
        resulting_force_success = resulting_force_success.unsqueeze(0)
        all_force_applied = all_force_applied.unsqueeze(0)

        all_keypoints = torch.tensor(all_keypoints, requires_grad=True)

        output = {
            'keypoints': all_keypoints,
            'rotation': resulting_rotation,
            'position': resulting_position,
            'force_success_flag': resulting_force_success,
            'force_applied': all_force_applied,
            'force_direction': forces_directions,  # batch size x seq len -1 x number of cp x 3
        }

        return output, target

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
