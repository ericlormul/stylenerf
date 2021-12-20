import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import misc
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import ToRGBLayer, MappingNetwork, SynthesisLayer
from nerf_sample_ray_split import RaySamplerSingleImage 
from camera import get_camera_mat, get_random_pose, get_camera_pose

# Util function for loading meshes
from pytorch3d.io import load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    camera_position_from_spherical_angles,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardFlatShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointLights
)

from collections import OrderedDict
import random

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out       

class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        up                      = 1,            # upsample factor
        to_rgb                  = True,          # early nerf stage won't output rgb img
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.up = up
        self.to_rgb = to_rgb

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=up,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if to_rgb and (is_last or architecture == 'skip'):
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=up,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            if self.to_rgb:
                misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if self.to_rgb:
            if img is not None and self.up > 1:
                misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
                img = upfirdn2d.upsample2d(img, self.resample_filter)
            if self.is_last or self.architecture == 'skip':
                y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real

class NeRFNet(nn.Module):
    def __init__(self,
        max_freq_log2,              # positional embedding
        max_freq_log2_viewdirs,

        w_dim,
        fg_netdepth,
        fg_dim,
        bg_netdepth,
        bg_dim,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__()
        
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=max_freq_log2 - 1,
                                             N_freqs=max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=max_freq_log2_viewdirs - 1,
                                            N_freqs=max_freq_log2_viewdirs)

        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=max_freq_log2 - 1,
                                             N_freqs=max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=max_freq_log2_viewdirs - 1,
                                            N_freqs=max_freq_log2_viewdirs)      
        self.fg_netdepth = fg_netdepth
        self.bg_netdepth = bg_netdepth


        self.num_ws = 0

        in_channels = self.fg_embedder_position.out_dim
        out_channels = fg_dim
        self.fg_net = []
        for i in range(fg_netdepth):
            is_last = (i == fg_netdepth - 1)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=32,
                img_channels=3, is_last=is_last, use_fp16=False, to_rgb=False, kernel_size=1, use_noise=False, **block_kwargs)
            self.num_ws += block.num_conv
            self.fg_net.append(( f'fg{i}', block))
            in_channels = out_channels
        self.fg_net = nn.ModuleDict(self.fg_net)     
        self.fg_sigma_layer = nn.Conv2d(out_channels, 1, kernel_size=1 )  
        

        in_channels = self.bg_embedder_position.out_dim
        out_channels = bg_dim
        self.bg_net = []
        for i in range(bg_netdepth):
            is_last = (i == bg_netdepth - 1)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=32,
                img_channels=3, is_last=is_last, use_fp16=False, to_rgb=False, kernel_size=1, use_noise=False, **block_kwargs)
            self.num_ws += block.num_conv
            self.bg_net.append(( f'bg{i}', block))
            in_channels = out_channels   
        self.bg_net = nn.ModuleDict(self.bg_net)
        self.bg_sigma_layer = nn.Conv2d(out_channels, 1, kernel_size=1 ) 
    def forward(self, ws, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, **block_kwargs):
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm      # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        x = self.fg_embedder_position(fg_pts)
        
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for i in range(self.fg_netdepth):
            block = getattr(self.fg_net, f'fg{i}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv   

        img = None
        x = x.permute(0,3,1,2)
        for i, cur_ws in zip(range(self.fg_netdepth), block_ws):
            block = getattr(self.fg_net, f'fg{i}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        x = x.permute(0,2,3,1)
        fg_raw = {}
        fg_raw['sigma'] = self.fg_sigma_layer(x.permute(0,3,1,2)).permute(0,2,3,1)
        fg_raw['sigma'] = torch.abs(fg_raw['sigma']).squeeze(-1)
        fg_raw['feature'] = x
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]), dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T     # [..., N_samples]
        fg_f_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['feature'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        x = self.bg_embedder_position(bg_pts)
        # near_depth: physical far; far_depth: physical near
        x = torch.flip(x, dims=[-2,])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]

        block_ws = []
        for i in range(self.bg_netdepth):
            block = getattr(self.bg_net, f'bg{i}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv   

        img = None
        x = x.permute(0,3,1,2)
        for i, cur_ws in zip(range(self.bg_netdepth), block_ws):
            block = getattr(self.bg_net, f'bg{i}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        x = x.permute(0,2,3,1)    
        # sigma layer 
        bg_raw = {}
        bg_raw['sigma'] = self.bg_sigma_layer(x.permute(0,3,1,2)).permute(0,2,3,1)
        bg_raw['sigma'] = torch.abs(bg_raw['sigma']).squeeze(-1)
        bg_raw['feature'] = x

        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_f_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['feature'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        
        # composite foreground and background
        bg_f_map = bg_lambda.unsqueeze(-1) * bg_f_map
        bg_depth_map = bg_lambda * bg_depth_map
        composite_map = fg_f_map + bg_f_map

        ret = OrderedDict([('composite_map', composite_map),            # loss
                           ('fg_weights', fg_weights),                  # importance sampling
                           ('bg_weights', bg_weights),                  # importance sampling
                           ('fg_f', fg_f_map),                      # below are for logging
                           ('fg_depth', fg_depth_map),
                           ('bg_f', bg_f_map),
                           ('bg_depth', bg_depth_map),
                           ('bg_lambda', bg_lambda)])  

        return ret

def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # TODO what is d1 and d2? and what is ray_d * ray_o
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples             

class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(6, self.img_resolution_log2 + 1)] # TODO 32->64
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 64 else 512
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, up=2, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, x, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])        

class StyleNeRF(nn.Module):
    def __init__(self,
        plane_H,
        plane_W,
        cascade_level,
        cascade_samples,
        use_single_nerf,
        **nerf_kwargs,
    ):
        super().__init__()          
        # nerf setup
        self.cascade_level = cascade_level
        self.cascade_samples = [int(x.strip()) for x in cascade_samples.split(',')]

        self.nerf_netdepth = nerf_kwargs['fg_netdepth'] + nerf_kwargs['bg_netdepth']
        self.nerf_models = self.create_nerf(cascade_level=cascade_level, use_single_nerf=use_single_nerf, nerf_kwargs=nerf_kwargs)

        # camera setup
        self.H = plane_H
        self.W = plane_W

        self.camera_intrinsic = get_camera_mat(fov=10, res=(self.W, self.H))
        # self.range_u, self.range_v = [0., 0.5], [1e-9, 0.001] # control camera postion on a sphere, 0 for range_v will canuse singular c2w matrix which is not inversable.
        # self.range_u, self.range_v = [0., 0.], [0.4167, 0.5]
        # self.range_u_0, self.range_v_0 = [0., 0.04], [0.5, 0.5]
        # self.range_u_1, self.range_v_1 = [0.96, 1.], [0.5, 0.5]
        self.range_u, self.range_v = [0., 0.083], [0.5, 0.5]
        self.range_radius = [1, 1]      # scales camera position with sphere radius

        # stylgan synthesis  
        self.remap_nerf_to_stylgan = nn.Conv2d(256, 512, kernel_size=1)

        self.synthesis = SynthesisNetwork(
            w_dim = 512,                 # Intermediate latent (W) dimensionality.
            img_resolution=256,             # Output image resolution.
            img_channels=3,               # Number of color channels.             
        )        
        self.num_ws = self.synthesis.num_ws + self.nerf_models['net_0'].num_ws

        self.sampled_batch_camera_loc = None
    def create_nerf(self, cascade_level, use_single_nerf, nerf_kwargs):
        models = nn.ModuleDict()
        net = None
        for m in range(cascade_level):
            if (use_single_nerf) and (net is not None):
                net = net
            else:
                net = NeRFNet(**nerf_kwargs)
            models['net_{}'.format(m)] = net
        return models    

    def forward(self, ws, **kwargs):
        device = ws.device
        range_u_type = 'range'
        range_v_type = 'range'
        if 'at_inference' in kwargs and kwargs['at_inference'] and ws.shape[0] == 1:
            ws = ws.repeat(32, 1, 1)
            range_u_type = 'linespace'
            range_v_type = 'range'
        batch_size = ws.shape[0]
        # poses, loc = get_random_pose(self.range_u, self.range_v, self.range_radius, range_u_type='points', range_v_type='linespace', batch_size=batch_size)

        poses, loc = get_random_pose(self.range_u, self.range_v, self.range_radius, range_u_type=range_u_type, range_v_type=range_v_type, batch_size=batch_size)

        self.sampled_batch_camera_loc = loc

        img_raysamplers = []
        for i in range(len(poses)):
            img_raysamplers.append(
                RaySamplerSingleImage(
                    H=self.H,
                    W=self.W,
                    intrinsics=self.camera_intrinsic,
                    c2w=poses[i].reshape(4,4)
                )   
            )   
              
        # randomly sample rays and move to device
        ray_batch = img_raysamplers[0].random_sample()
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].unsqueeze(0)
        ray_batches = [img_raysampler.random_sample() for img_raysampler in img_raysamplers[1:]]
        for rb in ray_batches:
            for key in rb:
                if torch.is_tensor(rb[key]):
                    ray_batch[key] = torch.cat([ray_batch[key], rb[key].unsqueeze(0)], dim=0)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):  
                ray_batch[key] = ray_batch[key].to(device)

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        for m in range(self.cascade_level):
            net = self.nerf_models['net_{}'.format(m)]

            # sample depths
            N_samples = self.cascade_samples[m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                fg_near_depth = ray_batch['min_depth']  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                fg_depth = perturb_samples(fg_depth)   # random perturbation during training

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples, device=ws.device).view(
                            [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,])
                bg_depth = perturb_samples(bg_depth)   # random perturbation during training
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                            N_samples=N_samples, det=False)    # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                            N_samples=N_samples, det=False)    # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))
            ret = net(ws, ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth)
        
        # stylgan synthesis
        ws = ws[:,self.nerf_models['net_0'].num_ws:]
        x = ret['composite_map'].reshape(-1, self.H, self.W, ret['composite_map'].shape[-1]).permute(0,3,1,2)
        x = self.remap_nerf_to_stylgan(x)
        imgs  = self.synthesis(x, ws)      
        return imgs

class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = StyleNeRF(
            w_dim=w_dim,

            plane_H=32,
            plane_W=32,
            cascade_level=2,
            cascade_samples='64,64',

            fg_netdepth=4,
            fg_dim=256,
            bg_netdepth=2,
            bg_dim=256,
            max_freq_log2=10,
            max_freq_log2_viewdirs=4,
            use_single_nerf=True        

        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
    def render_mesh(
            self,
            obj_path,
            batch_size, # Set batch size - this is the number of different viewpoints from which we want to render the mesh.
            device
        ):
        # Get vertices, faces, and auxiliary information:
        verts, faces, aux = load_obj(
            obj_path,
            device=device
            )        

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]*0.7  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=textures) 

        # Create a batch of meshes by repeating the cow mesh and associated textures. 
        # Meshes has a useful `extend` method which allows us do this very easily. 
        # This also extends the textures. 
        meshes = mesh.extend(batch_size)

        # Get a batch of viewing angles. 
        elev = torch.linspace(0, 0, batch_size)
        azim = torch.linspace(-180, 180, batch_size)

        # All the cameras helper methods support mixed type inputs and broadcasting. So we can 
        # view the camera from the same distance and specify dist=2.7 as a float,
        # and then specify elevation and azimuth angles for each viewpoint as tensors. 
        R, T = look_at_view_transform(dist=0.4, elev=elev, azim=azim)
        import pdb; pdb.set_trace()
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.4)

        # Here we set the output image to be of size 256 x 256 based on config.json 
        raster_settings = RasterizationSettings(
            image_size = 256, 
            blur_radius = 0.0, 
            faces_per_pixel = 1, 
        )

        # Initialize rasterizer by using a MeshRasterizer class
        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            )        

        lights = PointLights(device=device, specular_color=[[0.2,0.2,0.2]],diffuse_color=[[0.5,0.5,0.5]], location=[[0.0, 0.0, 3.0]])
        shader = HardFlatShader(device = device, cameras = cameras, lights=lights)

        # Create a mesh renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(rasterizer, shader)            

        # Move the light back in front of the cow which is facing the -z direction.
        # lights.location = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        lights.location = camera_position_from_spherical_angles(0.7, elev, azim).to(device)

        # We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
        # so the renderer does not need to be reinitialized if any of the settings change.
        images = renderer(meshes, cameras=cameras, lights=lights).detach()
        images = images[..., :3]

        return images, cameras


if __name__ == "__main__":
    G = Generator(
        z_dim=512,                      # Input latent (Z) dimensionality.
        c_dim=0,                      # Conditioning label (C) dimensionality.
        w_dim=512,                      # Intermediate latent (W) dimensionality.
        img_resolution=256,             # Output resolution.
        img_channels=3,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
    )

    z = torch.randn(2, 512)
    G.forward(z=z, c=None)