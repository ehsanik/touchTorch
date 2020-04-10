import torch
import torch.nn as nn

from utils.environment_util import EnvState
from utils.projection_utils import get_keypoint_projection, get_all_objects_keypoint_tensors
from .base_model import BaseModel
from utils.net_util import input_embedding_net, EnvWHumanCpFiniteDiffFast, combine_block_w_do

from torchvision.models.resnet import resnet18
from solvers import metrics


class BaselineRegressForce(BaseModel):
    metric = [
    ]

    def __init__(self, args):
        super(BaselineRegressForce, self).__init__(args)
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
        self.input_feature_size = self.object_feature_size
        self.cp_feature_size = self.number_of_cp * 3

        self.image_embed = combine_block_w_do(512, 64, args.dropout_ratio)
        input_object_embed_size = torch.Tensor([3 + 4, 100, self.object_feature_size])
        self.input_object_embed = input_embedding_net(input_object_embed_size.long().tolist(), dropout=args.dropout_ratio)
        contact_point_embed_size = torch.Tensor([3 * 5, 100, self.object_feature_size])
        self.contact_point_embed = input_embedding_net(contact_point_embed_size.long().tolist(), dropout=args.dropout_ratio)
        self.lstm_encoder = nn.LSTM(input_size=64 * 7 * 7 + 512, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)

        force_decoder_size = torch.Tensor([self.hidden_size * 2, 100, (3) * self.number_of_cp])
        self.force_decoder = input_embedding_net(force_decoder_size.long().tolist(), dropout=args.dropout_ratio)
        assert (args.mode == 'train' and args.batch_size > 1 and args.break_batch == 1) or (args.mode != 'train' and args.batch_size == 1)

        self.train_mode = (args.mode == 'train')

        assert self.number_of_cp == 5  # for five fingers

        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

        if not self.train_mode:
            BaselineRegressForce.metric += [
                metrics.ObjKeypointMetric,  # During test time add it
                metrics.ObjRotationMetric,
                metrics.ObjPositionMetric,
            ]

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
        return x

    def forward(self, input, target):

        initial_position = input['initial_position']
        initial_rotation = input['initial_rotation']
        rgb = input['rgb']
        batch_size, seq_len, c, w, h = rgb.shape
        contact_point_as_input = input['contact_points'].view(batch_size, 5 * 3)

        image_features = self.resnet_features(rgb)
        image_features = self.image_embed(image_features.view(batch_size * seq_len, 512, 7, 7)).view(batch_size, seq_len, 64 * 7 * 7)

        initial_object_features = self.input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
        object_features = initial_object_features.unsqueeze(1).repeat(1, self.sequence_length, 1)  # add a dimension for sequence length and then repeat that

        input_embedded = torch.cat([image_features, object_features], dim=-1)
        embedded_sequence, (hidden, cell) = self.lstm_encoder(input_embedded)

        contact_point_embedding = self.contact_point_embed(contact_point_as_input).unsqueeze(1).repeat(1, seq_len, 1)
        combined_w_cp = torch.cat([embedded_sequence, contact_point_embedding], dim=-1)
        forces_prediction = self.force_decoder(combined_w_cp).view(batch_size, seq_len, self.number_of_cp, 3)  # Predict contact point for each image
        forces_prediction = forces_prediction[:, 1:, :, :]

        forces_prediction = forces_prediction.clamp(-1.5, 1.5)

        output = {
            'forces': forces_prediction,  # batchsize x seq len x number of cp x 3
        }
        target['object_name'] = input['object_name']

        # forces_prediction[:] = 0.
        # output['forces'][:] = 0.

        #  remove
        # forces_prediction[:] = target['forces']

        if not self.train_mode:
            object_name = input['object_name']
            assert len(object_name) == 1
            object_name = object_name[0]

            contact_points = input['contact_points']
            assert contact_points.shape[0] == 1
            contact_points = contact_points.squeeze(0)
            resulting_position = []
            resulting_rotation = []

            env_state = EnvState(object_name=object_name, rotation=initial_rotation[0], position=initial_position[0], velocity=None, omega=None)
            for seq_ind in range(self.sequence_length - 1):
                force = forces_prediction[0, seq_ind].view(self.number_of_cp * 3)
                # initial initial_velocity is whatever it was the last frame, note that the gradients are not backproped here
                env_state, force_success, force_applied = self.environment_layer.apply(self.environment, env_state.toTensor(), force, contact_points)
                env_state = EnvState.fromTensor(env_state)

                resulting_position.append(env_state.position)
                resulting_rotation.append(env_state.rotation)
            resulting_position = torch.stack(resulting_position, dim=0)
            resulting_rotation = torch.stack(resulting_rotation, dim=0)
            resulting_position = resulting_position.unsqueeze(0)  # adding batchsize back because we need it in the loss
            resulting_rotation = resulting_rotation.unsqueeze(0)  # adding batchsize back because we need it in the loss

            all_keypoints = get_keypoint_projection(object_name, resulting_position, resulting_rotation, self.all_objects_keypoint_tensor[object_name])
            all_keypoints = all_keypoints.unsqueeze(0)  # adding batchsize back because we need it in the loss
            output['keypoints'] = all_keypoints
            output['rotation'] = resulting_rotation
            output['position'] = resulting_position

            output['force_applied'] = output['forces']
            output['force_direction'] = output['forces']

        return output, target

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
