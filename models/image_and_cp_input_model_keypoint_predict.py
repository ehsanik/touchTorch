import torch
import torch.nn as nn
from .base_model import BaseModel
from utils.net_util import input_embedding_net, EnvWHumanCpFiniteDiffFast, combine_block_w_do

from torchvision.models.resnet import resnet18
from solvers import metrics
from utils.environment_util import EnvState
from utils.projection_utils import get_keypoint_projection, get_all_objects_keypoint_tensors


class ImageAndCPInputKPOutModel(BaseModel):
    metric = [
        metrics.ObjRotationMetric,
        metrics.ObjPositionMetric,
        metrics.ObjKeypointMetric,
    ]

    def __init__(self, args):
        super(ImageAndCPInputKPOutModel, self).__init__(args)
        self.environment_layer = EnvWHumanCpFiniteDiffFast
        self.loss_function = args.loss
        self.relu = nn.LeakyReLU()
        self.number_of_cp = args.number_of_cp
        self.environment = args.instance_environment
        self.sequence_length = args.sequence_length
        self.gpu_ids = args.gpu_ids

        self.feature_extractor = resnet18(pretrained=args.pretrain)
        del self.feature_extractor.fc

        self.feature_extractor.eval()

        self.image_feature_size = 512
        self.object_feature_size = 512
        self.hidden_size = 512
        self.num_layers = 3
        # self.input_feature_size = self.image_feature_size + self.object_feature_size
        self.input_feature_size = self.object_feature_size
        self.cp_feature_size = self.number_of_cp * 3

        self.image_embed = combine_block_w_do(512, 64, args.dropout_ratio)

        input_object_embed_size = torch.Tensor([3 + 4, 100, self.object_feature_size])
        self.input_object_embed = input_embedding_net(input_object_embed_size.long().tolist(), dropout=args.dropout_ratio)
        state_embed_size = torch.Tensor([EnvState.total_size + self.cp_feature_size, 100, self.object_feature_size])
        self.state_embed = input_embedding_net(state_embed_size.long().tolist(), dropout=args.dropout_ratio)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size + 64 * 7 * 7, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)

        self.lstm_decoder = nn.LSTMCell(input_size=self.hidden_size * 2, hidden_size=self.hidden_size)

        forces_directions_decoder_size = torch.Tensor([self.hidden_size, 100, (3) * self.number_of_cp])

        self.forces_directions_decoder = input_embedding_net(forces_directions_decoder_size.long().tolist(), dropout=args.dropout_ratio)
        self.force_clamping = args.force_clamping

        assert args.batch_size == 1, 'have not been implemented yet, because of the environment'

        assert self.number_of_cp == 5  # for five fingers

        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

    def loss(self, args):
        return self.loss_function(args)

    def resnet_features(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            batch_size, seq_len, c, w, h = x.shape
            x = x.view(batch_size * seq_len, c, w, h)
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)

            x = self.feature_extractor.layer1(x)
            x = self.feature_extractor.layer2(x)
            x = self.feature_extractor.layer3(x)
            x = self.feature_extractor.layer4(x)

            x = x.view(batch_size, seq_len, 512, 7, 7)
            x = x.detach()

        return x


    def forward(self, input, target):

        initial_position = input['initial_position']
        initial_rotation = input['initial_rotation']
        rgb = input['rgb']
        contact_points = input['contact_points']
        assert contact_points.shape[0] == 1
        contact_points = contact_points.squeeze(0)

        contact_point_as_input = input['contact_points'].view(1, 3 * 5) # Batchsize , 3 * number of contact points
        batch_size, seq_len, c, w, h = rgb.shape


        image_features = self.resnet_features(rgb)
        image_features = self.image_embed(image_features.view(batch_size * seq_len, 512, 7, 7)).view(batch_size, seq_len, 64 * 7 * 7)

        object_name = input['object_name']
        assert len(object_name) == 1
        object_name = object_name[0]

        initial_object_features = self.input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
        object_features = initial_object_features.unsqueeze(1).repeat(1, self.sequence_length, 1) #add a dimension for sequence length and then repeat that

        input_embedded = torch.cat([image_features, object_features], dim=-1)
        embedded_sequence, (hidden, cell) = self.lstm_encoder(input_embedded)


        last_hidden = hidden.view(self.num_layers, 1, 1, self.hidden_size)[-1, -1, :, :] #num_layers, num direction, batchsize, hidden and then take the last hidden layer
        last_cell = cell.view(self.num_layers, 1, 1, self.hidden_size)[-1, -1, :, :]  # num_layers, num direction, batchsize, cell and then take the last hidden layer

        hn = last_hidden
        cn = last_cell

        resulting_force_success = []
        forces_directions = []
        all_force_applied = []

        env_state = EnvState(object_name=object_name, rotation=initial_rotation[0], position=initial_position[0], velocity=None, omega=None)
        resulting_position = []
        resulting_rotation = []

        for seq_ind in range(self.sequence_length - 1):
            prev_location = env_state.toTensorCoverName().unsqueeze(0)

            prev_state_and_cp = torch.cat([prev_location, contact_point_as_input], dim=-1)
            prev_state_embedded = self.state_embed(prev_state_and_cp)
            next_state_embedded = embedded_sequence[:, seq_ind + 1]
            input_lstm_cell = torch.cat([prev_state_embedded, next_state_embedded], dim=-1)

            (hn, cn) = self.lstm_decoder(input_lstm_cell, (hn, cn))
            force = self.forces_directions_decoder(hn)
            assert force.shape[0] == 1
            force = force.squeeze(0)
            assert force.shape[0] == (self.number_of_cp * 3)

            # Cap forces
            # Prevent force from diverging
            force = force.clamp(-self.force_clamping, self.force_clamping)


            # initial initial_velocity is whatever it was the last frame, note that the gradients are not backproped here
            env_state, force_success, force_applied = self.environment_layer.apply(self.environment, env_state.toTensor(), force,contact_points)
            env_state = EnvState.fromTensor(env_state)

            resulting_position.append(env_state.position)
            resulting_rotation.append(env_state.rotation)
            resulting_force_success.append(force_success)
            forces_directions.append(force.view(self.number_of_cp, 3))
            all_force_applied.append(force_applied)


        resulting_position = torch.stack(resulting_position, dim=0)
        resulting_rotation = torch.stack(resulting_rotation, dim=0)
        resulting_force_success = torch.stack(resulting_force_success, dim=0)
        forces_directions = torch.stack(forces_directions, dim=0)
        all_force_applied = torch.stack(all_force_applied, dim=0)

        resulting_position = resulting_position.unsqueeze(0)  # adding batchsize back because we need it in the loss
        resulting_rotation = resulting_rotation.unsqueeze(0)  # adding batchsize back because we need it in the loss
        forces_directions = forces_directions.unsqueeze(0)
        resulting_force_success = resulting_force_success.unsqueeze(0)
        all_force_applied = all_force_applied.unsqueeze(0)

        all_keypoints = get_keypoint_projection(object_name, resulting_position, resulting_rotation, self.all_objects_keypoint_tensor[object_name])
        all_keypoints = all_keypoints.unsqueeze(0)  # adding batchsize back because we need it in the loss

        output = {
            'keypoints': all_keypoints,
            'rotation': resulting_rotation,
            'position': resulting_position,
            'force_success_flag': resulting_force_success,
            'force_applied': all_force_applied,
            'force_direction': forces_directions,  # batch size x seq len -1 x number of cp x 3
        }

        target['object_name'] = input['object_name']


        return output, target

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
