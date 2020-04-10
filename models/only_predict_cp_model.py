import torch
import torch.nn as nn

from utils.projection_utils import get_all_objects_keypoint_tensors
from .base_model import BaseModel
from utils.net_util import input_embedding_net, EnvWHumanCpFiniteDiffFast, combine_block_w_do

from torchvision.models.resnet import resnet18
from solvers import metrics


class NoForceOnlyCPModel(BaseModel):
    metric = [
        metrics.CPMetric,
    ]

    def __init__(self, args):
        super(NoForceOnlyCPModel, self).__init__(args)
        self.environment_layer = EnvWHumanCpFiniteDiffFast
        self.loss_function = args.loss
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
        self.lstm_encoder = nn.LSTM(input_size=64 * 7 * 7 + 512, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)

        contact_point_decoder_size = torch.Tensor([self.hidden_size, 100, (3) * self.number_of_cp])
        self.contact_point_decoder = input_embedding_net(contact_point_decoder_size.long().tolist(), dropout=args.dropout_ratio)

        assert self.number_of_cp == 5  # for five fingers
        self.all_objects_keypoint_tensor = get_all_objects_keypoint_tensors(args.data)
        if args.gpu_ids != -1:
            for obj, val in self.all_objects_keypoint_tensor.items():
                self.all_objects_keypoint_tensor[obj] = val.cuda()

    def loss(self, args):
        return self.loss_function(args)

    def resnet_features(self, x):  # change this to groupnorm and leaky relu maybe?
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

        image_features = self.resnet_features(rgb)
        image_features = self.image_embed(image_features.view(batch_size * seq_len, 512, 7, 7)).view(batch_size, seq_len, 64 * 7 * 7)
        initial_object_features = self.input_object_embed(torch.cat([initial_position, initial_rotation], dim=-1))
        object_features = initial_object_features.unsqueeze(1).repeat(1, self.sequence_length, 1)  # add a dimension for sequence length and then repeat that

        input_embedded = torch.cat([image_features, object_features], dim=-1)
        embedded_sequence, (hidden, cell) = self.lstm_encoder(input_embedded)

        contact_points_prediction = self.contact_point_decoder(embedded_sequence).view(batch_size, seq_len, self.number_of_cp, 3)  # Predict contact point for each image

        # Use the last prediction for the whole sequence, last prediction because we need to see the whole sequence to come to a conclusion
        contact_points_prediction = contact_points_prediction[:, -1:].repeat(1, seq_len, 1, 1)

        output = {
            'contact_points': contact_points_prediction,  # batchsize x seq len x number of cp x 3
        }

        target['object_name'] = input['object_name']

        return output, target

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
