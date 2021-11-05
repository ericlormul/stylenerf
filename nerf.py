from types import new_class
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy
from torch_utils import misc
from training import networks_stylegan3
from camera import get_camera_mat, get_random_pose, get_camera_pose
from nerf_sample_ray_split import RaySamplerSingleImage
from training.networks_stylegan2 import SynthesisNetwork   
from nerf_network import SynthesisNetwork   

class Generator(torch.nn.Module):
    def __init__(self,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = synthesis_kwargs['z_dim']
        self.c_dim = synthesis_kwargs['c_dim']
        self.w_dim = synthesis_kwargs['w_dim']
        self.img_resolution = synthesis_kwargs['img_resolution']

        self.synthesis = SynthesisNetwork(synthesis_kwargs)
        self.num_ws = synthesis_kwargs['n_styled_conv_layers']
        self.mapping = networks_stylegan3.MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws)
        return img

if __name__ == "__main__":
    if False:
        mapping = networks_stylegan3.MappingNetwork(
            z_dim=16, c_dim=0, w_dim=32, num_ws=8
        )
        z = torch.randn(2, 16)
        print(mapping(z, None).shape)

        fg_embedder_position = Embedder(input_dim=3,
                                        max_freq_log2=10 - 1,
                                        N_freqs=10)    
        import pdb; pdb.set_trace()
        print(fg_embedder_position(torch.randn(2,3)).shape)

        camera_intrinsic = get_camera_mat(fov=10)
        range_u, range_v = [0, 0], [0.4167, 0.5]
        range_radius = [2.732, 2.732]
        pose = get_random_pose(range_u, range_v, range_radius, batch_size=1)

        import pdb; pdb.set_trace()
        img_raysampler = RaySamplerSingleImage(H=64, W=64, intrinsics=camera_intrinsic, c2w=pose.reshape(4,4),
                                                        img_path=None,
                                                        mask_path=None,
                                                        min_depth_path=None,
                                                        max_depth=None)
        ret = img_raysampler.random_sample(N_rand=1024)                                                    
        import pdb; pdb.set_trace()
    synthesis_args = {
        #nerf configs
        'cascade_level': 2,
        'cascade_samples': '64,64',
        #convnet configs
        'w_dim': 512,
        'fg_netdepth': 4,
        'bg_netdepth': 2,
        'upsampling_netdepth': 4,
        'n_styled_conv_layers': 11, # fg_netdepth + bg_netdepth + upsampling_netdepth + 1  (toRGB),
        'conv_out_channels': 256,
        'up_out_channels': 128,
        'use_viewdirs': False,          # not used

        # embedder configs
        'max_freq_log2': 10,
        'max_freq_log2_viewdirs': 4,

        # ray sampler configs
        'plane_H': 64,
        'plane_W': 64,
        'N_rand': 1024 # not used, N_rand should be H*W all pixel positions
    }
    from argparse import Namespace
    synthesis_args = Namespace(**synthesis_args)

    syn_net = SynthesisNetwork(
        args=synthesis_args,
    )
    ws = torch.randn(32, 11, 512)
    syn_net(ws)