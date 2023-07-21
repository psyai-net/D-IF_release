# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):

    def __init__(self,
                 filter_channels,
                 name=None,
                 res_layers=[],
                 norm='group',
                 last_op=None,
                 mode=None):

        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = res_layers    # [2,3,4]
        self.norm = norm
        self.last_op = last_op
        self.mode = mode
        self.name = name
        self.activate = nn.LeakyReLU(inplace=True)
        filter_channels = [13, 512, 256, 128, 2]
        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0],
                              filter_channels[l + 1], 1))
            else:
                self.filters.append(
                    nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch': # in
                    self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm1d(filter_channels[l +
                                                                        1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l],
                                                           name='weight')
        self.filters_fine = nn.ModuleList()
        self.norms_fine = nn.ModuleList()
        filter_channels_fine = [16, 512, 256, 128, 1]
        for l in range(0, len(filter_channels_fine) - 1):
            if l in self.res_layers:
                self.filters_fine.append(
                    nn.Conv1d(filter_channels_fine[l] + filter_channels_fine[0],
                              filter_channels_fine[l + 1], 1))
            else:
                self.filters_fine.append(
                    nn.Conv1d(filter_channels_fine[l], filter_channels_fine[l + 1], 1))
            if l != len(filter_channels_fine) - 2:
                self.norms_fine.append(nn.BatchNorm1d(filter_channels_fine[l + 1]))
    # todo: yxt need to change
    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):

            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))

        mu_0, sigma_0 = torch.split(y, 1, dim=1) # (B, C=1, N), (B, C=1, N)
        sigma_0 = F.softplus(sigma_0)
        sigma_0 += 1e-8
        q_distribution = torch.distributions.Normal(mu_0, sigma_0)
        z = q_distribution.rsample()

        if self.mode != 'test':
            feat_fine = torch.cat([tmpy, z, mu_0, sigma_0], 1)  # (B, C=13+2+1, N)
        else:
            feat_fine = torch.cat([tmpy, mu_0, mu_0, sigma_0], 1)
        fine_y = feat_fine
        tmp_fine_y = feat_fine

        for i, f in enumerate(self.filters_fine):
            fine_y = f(fine_y if i not in self.res_layers else torch.cat([fine_y, tmp_fine_y], 1))
            if i != len(self.filters_fine) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    fine_y = self.activate(fine_y)
                else:
                    fine_y = self.activate(self.norms_fine[i](fine_y)) # (B, C=1, N)

        if self.last_op is not None:
            fine_y = self.last_op(fine_y) # (B, C=1, N)
        else:
            fine_y = fine_y
        if self.mode != 'test':
            return fine_y, mu_0, sigma_0
        else:
            return fine_y

if __name__ == "__main__":
    from torchsummary import summary
    encoder = MLP(
            filter_channels=[13, 512, 256, 128, 1],
            name="if",
            res_layers=[2,3,4],
            norm='batch',
            last_op=nn.Sigmoid(),
        )
    # print(encoder)

    # total_num = sum(p.numel() for p in encoder.parameters())
    # trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # print('Total', total_num, 'Trainable', trainable_num)
    # input = torch.ones(8, 13, 888)
    # out = encoder(input)
    # print(out.shape)
    
    summary(encoder.to('cuda'), input_size=( 13, 888), batch_size=-1)