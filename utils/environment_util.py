import torch
import numpy as np
import torch.nn.functional as F
from utils.constants import ALL_OBJECTS

REGISTERED_OBJECTS = ALL_OBJECTS


def convert_obj_name_to_tensor(object_name):
    return torch.Tensor([REGISTERED_OBJECTS.index(object_name)]).float()


def convert_tensor_to_obj_name(object_name_tensor):
    object_ind = object_name_tensor.item()
    int_object_ind = int(object_ind)
    assert (int_object_ind - object_ind) <= 0.01
    assert int_object_ind < len(REGISTERED_OBJECTS)
    return REGISTERED_OBJECTS[int_object_ind]


class EnvState:
    size = [3, 4, 3, 3, 1]
    total_size = sum(size)
    OBJECT_TYPE_INDEX = total_size - 1

    def __init__(self, object_name, position, rotation, velocity=None, omega=None, device=None):
        if velocity is None:
            velocity = torch.tensor([0., 0., 0.], device=position.device, requires_grad=True)
        if omega is None:
            omega = torch.tensor([0., 0., 0.], device=position.device, requires_grad=True)

        assert len(position) == 3 and len(rotation) == 4 and len(velocity) == 3 and len(omega) == 3

        [position, rotation, velocity, omega] = [convert_to_tensor(x) for x in [position, rotation, velocity, omega]]
        if not device:
            [position, rotation, velocity, omega] = [x.to(device) for x in [position, rotation, velocity, omega]]

        # ((1+0.01*z)/(1+2z*0.01+0.01^2)) -> function of diff of (w,x,y,z) - (w,x,y,z+eps)
        rotation = F.normalize(rotation.unsqueeze(0)).squeeze(0)

        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.omega = omega
        self.object_name = object_name

    def toTensor(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = convert_obj_name_to_tensor(self.object_name)
        object_name_tensor = object_name_tensor.to(self.position.device)
        tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega, object_name_tensor], dim=-1)
        assert tensor.shape[0] == EnvState.total_size
        return tensor

    @staticmethod
    def fromTensor(tensor):
        assert tensor.shape[0] == EnvState.total_size and len(tensor.shape) == 1
        position = tensor[0:3]
        rotation = tensor[3:7]
        velocity = tensor[7:10]
        omega = tensor[10:13]
        assert 0 <= tensor[EnvState.OBJECT_TYPE_INDEX] < len(REGISTERED_OBJECTS)
        object_name = convert_tensor_to_obj_name(tensor[EnvState.OBJECT_TYPE_INDEX])
        return EnvState(object_name, position, rotation, velocity, omega)

    def toTensorCoverName(self):
        assert type(self.position) == torch.Tensor
        assert type(self.rotation) == torch.Tensor
        assert type(self.velocity) == torch.Tensor
        assert type(self.omega) == torch.Tensor
        assert self.object_name in REGISTERED_OBJECTS
        object_name_tensor = torch.Tensor([-1.0]).float().to(self.position.device)
        tensor = torch.cat([self.position, self.rotation, self.velocity, self.omega, object_name_tensor], dim=-1)
        assert tensor.shape[0] == EnvState.total_size
        return tensor

    def clone(self):
        return EnvState(object_name=self.object_name, position=self.position.clone().detach(), rotation=self.rotation.clone().detach(), velocity=self.velocity.clone().detach(), omega=self.omega.clone().detach())

    def __str__(self):
        return 'object_name:{},position:{},rotation:{},velocity:{},omega:{}'.format(self.object_name, self.position, self.rotation, self.velocity, self.omega)

    def cuda_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cuda() for x in [self.position, self.rotation, self.velocity, self.omega]]

    def cpu_(self):
        [self.position, self.rotation, self.velocity, self.omega] = [x.cpu() for x in [self.position, self.rotation, self.velocity, self.omega]]


class ForceValOnly:
    size = [3]
    total_size = sum(size)

    def __init__(self, force, device=None):

        assert len(force) == 3

        force = convert_to_tensor(force)
        if not device:
            force = force.to(device)

        self.force = force

    @staticmethod
    def fromTensor(tensor):
        assert len(tensor.shape) == 1 and tensor.shape[0] == ForceValOnly.total_size
        force = tensor
        return ForceValOnly(force)

    def __str__(self):
        return 'force:{}'.format(self.force)

    def cpu_(self):
        self.force = self.force.cpu()

    @staticmethod
    def fromForceArray(force_array):
        force_array_shape = force_array.shape
        if len(force_array_shape) == 2:
            return [ForceValOnly.fromTensor(force_array[cp_ind]) for cp_ind in range(force_array_shape[0])]
        elif len(force_array_shape) == 3:
            return [ForceValOnly.fromForceArray(force_array[seq_ind]) for seq_ind in range(force_array_shape[0])]
        else:
            raise Exception('Not implemented')


def convert_to_tensor(x):
    if type(x) == tuple:
        result = torch.Tensor(x)
        result.requires_grad = True
        return result
    elif type(x) == torch.Tensor:
        return x
    else:
        import pdb;
        pdb.set_trace()
        raise Exception('Not implemented')


def convert_to_tuple(x):
    if type(x) == tuple:
        return x
    elif type(x) == torch.Tensor:
        x = x.cpu().tolist()
        return tuple(x)
    elif type(x) == np.ndarray:
        x = x.tolist()
        return tuple(x)
    else:
        import pdb;
        pdb.set_trace()
        raise Exception('Not implemented')
