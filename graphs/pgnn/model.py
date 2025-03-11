import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


##
# Code below extracted and adapted from P-GNN source code at
# https://github.com/JiaxuanYou/P-GNN/blob/master/model.py
##


class Nonlinear(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class PGNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.in_channels = in_channels
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, out_channels, 1)

        self.linear_hidden = nn.Linear(in_channels * 2, out_channels)
        self.linear_out_position = nn.Linear(out_channels, 1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages)  # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


class PGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        anchor_dim,
        out_channels,
        feature_pre=True,
        num_layers=2,
        dropout=0.5,
        **kwargs
    ):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            hidden_channels = out_channels
        if feature_pre:
            self.linear_pre = nn.Linear(in_channels, in_channels)
        self.convs = nn.ModuleList()
        self.init_conv = PGNN_layer(in_channels, hidden_channels)
        if num_layers > 1:
            self.convs.append(self.init_conv)
            for _ in range(num_layers - 2):
                self.convs.append(PGNN_layer(hidden_channels, hidden_channels))
            self.convs.append(PGNN_layer(hidden_channels, out_channels))
        else:
            self.convs.append(PGNN_layer(in_channels, out_channels))

        self.lin_out = nn.Linear(anchor_dim, out_channels)

    def forward(self, data):

        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        for i, conv in enumerate(self.convs):
            x_position, x = conv(x, data.dists_max, data.dists_argmax)

            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x_position = F.normalize(x_position, p=2, dim=-1)
        x_position = self.lin_out(x_position)

        return x_position
