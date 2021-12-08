import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace

# from https://github.com/blufzzz/learnable-triangulation-pytorch/blob/master/mvn/models/v2v.py


class SPADE(nn.Module):
    def __init__(self, 
                style_vector_channels, 
                features_channels, 
                hidden=64, # hidden=128
                ks=3):

        super().__init__()

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv3d(features_channels if STYLE_FORWARD else style_vector_channels, hidden, kernel_size=ks, padding=pw),
            get_activation(ACTIVATION_TYPE)()
        )

        self.gamma = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.beta = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.bn = nn.InstanceNorm3d(features_channels, affine=False)

    def forward(self, x, params):

        if params is None:
            return x

        batch_size = x.shape[0]

        params = F.interpolate(params, size=x.size()[2:], mode='trilinear')
        actv = self.shared(params)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        x = self.bn(x) * (1 + gamma) + beta

        return x 


class CompoundNorm(nn.Module):
    def __init__(self, normalization_types, out_planes, n_groups, style_vector_channels):
        super().__init__()
        norm_type, adaptive_norm_type = normalization_types
        assert norm_type in ORDINARY_NORMALIZATIONS and adaptive_norm_type in ADAPTIVE_NORMALIZATION
        self.norm = get_normalization(norm_type, out_planes, n_groups, style_vector_channels)
        self.adaptive_norm = get_normalization(adaptive_norm_type, out_planes, n_groups, style_vector_channels)
    def forward(self, x, params):
        
        x = self.norm(x)
        x = self.adaptive_norm(x, params)
        return x       


class GroupNorm(nn.Module):
    def __init__(self, n_groups, features_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(n_groups, features_channels)
    def forward(self, x, params=None):
        x = self.group_norm(x)
        return x      

class BatchNorm3d(nn.Module):
    def __init__(self, features_channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(features_channels, affine=False)
    def forward(self, x, params=None):
        x = self.batch_norm(x)
        return x           
        

def get_activation(activation_type):
    return {'LeakyReLU':nn.LeakyReLU,'ReLU':nn.ReLU}[activation_type]

def get_normalization(normalization_type, features_channels, n_groups=32, style_vector_channels=None):

    if type(normalization_type) is list:
        return CompoundNorm(normalization_type, features_channels, n_groups, style_vector_channels)
    else:    
        if normalization_type == 'adain':
            return AdaIN(style_vector_channels, features_channels, use_affine_mlp=False)
        if normalization_type == 'adain_mlp':
            return AdaIN(style_vector_channels, features_channels, use_affine_mlp=True)    
        elif normalization_type ==  'batch_norm':
            return BatchNorm3d(features_channels)
        elif normalization_type ==  'spade':
            return SPADE(style_vector_channels, features_channels)    
        elif normalization_type == 'group_norm':
            return GroupNorm(n_groups, features_channels)
        else:
            raise RuntimeError('{} is unknown normalization_type'.format(normalization_type))          
