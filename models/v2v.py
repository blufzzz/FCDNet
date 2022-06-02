# from https://github.com/blufzzz/learnable-triangulation-pytorch/blob/master/mvn/models/v2v.py

import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace


NORMALIZATION = 'batch_norm'
ACTIVATION = 'ReLU'


def normalization(out_planes):
    if NORMALIZATION == 'batch_norm':
        return nn.BatchNorm3d(out_planes, affine=False)
    elif NORMALIZATION == 'group_norm':
        return nn.GroupNorm(16, out_planes)
    elif NORMALIZATION == 'instance_norm':
        return nn.InstanceNorm3d(out_planes, affine=True)
    else:
        raise RuntimeError('Wrong NORMALIZATION')

def activation():
    if ACTIVATION == 'ReLU':
        return nn.ReLU(True)
    elif ACTIVATION == 'ELU':
        return nn.ELU()
    elif ACTIVATION == 'LeakyReLU':
        return nn.LeakyReLU()
    else:
        raise RuntimeError('Wrong ACTIVATION')



class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            normalization(out_planes),
            activation()
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            normalization(out_planes),
            activation(),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            normalization(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                normalization(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            normalization(out_planes),
            activation()
        )

    def forward(self, x):
        return self.block(x)



class EncoderDecorder(nn.Module):
    def __init__(self, max_channel=128):
        super().__init__()

        intermediate_channel = max(max_channel//2, 32)

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, intermediate_channel)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(intermediate_channel, max_channel)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(max_channel, max_channel)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(max_channel, max_channel)

        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(max_channel, max_channel)

        self.mid_res = Res3DBlock(max_channel, max_channel)

        self.decoder_res5 = Res3DBlock(max_channel, max_channel)
        self.decoder_upsample5 = Upsample3DBlock(max_channel, max_channel, 2, 2)

        self.decoder_res4 = Res3DBlock(max_channel, max_channel)
        self.decoder_upsample4 = Upsample3DBlock(max_channel, max_channel, 2, 2)
        self.decoder_res3 = Res3DBlock(max_channel, max_channel)
        self.decoder_upsample3 = Upsample3DBlock(max_channel, max_channel, 2, 2)
        self.decoder_res2 = Res3DBlock(max_channel, max_channel)
        self.decoder_upsample2 = Upsample3DBlock(max_channel, intermediate_channel, 2, 2)
        self.decoder_res1 = Res3DBlock(intermediate_channel, intermediate_channel)
        self.decoder_upsample1 = Upsample3DBlock(intermediate_channel, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(intermediate_channel, intermediate_channel)
        self.skip_res3 = Res3DBlock(max_channel, max_channel)
        self.skip_res4 = Res3DBlock(max_channel, max_channel)
        self.skip_res5 = Res3DBlock(max_channel, max_channel)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        skip_x4 = self.skip_res4(x)
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)

        skip_x5 = self.skip_res5(x)
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)

        x = self.mid_res(x)

        x = self.decoder_res5(x)
        x = self.decoder_upsample5(x)
        x = x + skip_x5 

        x = self.decoder_res4(x)
        x = self.decoder_upsample4(x)
        x = x + skip_x4
        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x

class V2VModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sigmoid = config.model.sigmoid
        input_channels = len(config.dataset.features)
        # add coordinates to model
        if hasattr(config.dataset, 'add_xyz') and config.dataset.add_xyz:
            input_channels += 3
        # add predicitons from another model 
        if hasattr(config.dataset, 'predictions_path'):
            input_channels += 1
        output_channels = config.model.output_channels
        max_channel = config.model.max_channel_encoder_decoder

        # nasty hack to replace normalization layers if the model
        global NORMALIZATION
        NORMALIZATION = config.model.normalization if hasattr(config.model,'normalization') else 'batch_norm'

        global ACTIVATION
        ACTIVATION = config.model.activation if hasattr(config.model,'activation') else 'ReLU'


        self.front_layers = nn.Sequential(
            # normalization(input_channels),
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            # Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder(max_channel)

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            # Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):

        # x - [bs, C, x,x,x]

        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)

        if self.sigmoid:
            return x.sigmoid()
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



