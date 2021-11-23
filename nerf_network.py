import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils.ops import filtered_lrelu
from training import networks_stylegan3
from nerf_sample_ray_split import RaySamplerSingleImage 
from camera import get_camera_mat, get_random_pose, get_camera_pose

# import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__package__)

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

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    b,                  # Biase tensor: [out_channels]
    s,                  # Style tensor: [batch_size, in_channels]
    dim         = 1,    # The dimension in `x` corresponding to the elements of `b`.
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]
    
    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    # x = F.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = F.conv2d(input=x, weight=w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
    return x

class Upsampling(nn.Module):
    def __init__(self,
        args
    ):
        super().__init__()
    
    def forward(self, x):
        pass

# conv1x1 synthesis blocks
class SynthesisBLK(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb = False,                       # Is this the final ToRGB layer?
        is_up = False,                           # Is this a upsampling block?

        # Input & output specifications.
        in_channels=256,                    # Number of input channels.
        out_channels=256,                   # Number of output channels.        
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_up = is_up
        self.in_channels = in_channels
        self.out_channels =  3 if self.is_torgb else out_channels
        self.conv_kernel = 1 

        # Setup parameters and buffers.
        self.affine = networks_stylegan3.FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))        

        if self.is_up:
            self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inputs): # single inputs tuple for nn.sequential, (x, w)
        x, w = inputs
        styles = self.affine(w)
        x = modulated_conv2d(x=x, w=self.weight, b=self.bias, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb))
        if self.is_up:
            x = self.upsampler(x)
        return x

class Conv1x1Net(nn.Module):
    def __init__(self,
        D=8,                    # D: network depth
        w_dim=128,
        in_channels=256,         # W: number of embedder channels
        out_channels=256,       # W: number of embedder channels
        input_ch_viewdirs=3,    # input_ch_viewdirs: input channels for encodings of view directions
        skips=[],              # skips: skip connection in network
        use_viewdirs=False,      # use_viewdirs: if True, will use the view directions as input
        is_bg = False
    ):
        super().__init__()

        self.in_channel = in_channels
        self.out_channels = out_channels
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        self.is_bg = is_bg

        self.base_layers = []
        dim = self.in_channel
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(
                    SynthesisBLK(w_dim=w_dim, in_channels=dim, out_channels=out_channels),
                    nn.LeakyReLU()
                )
            )
            dim = out_channels
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += in_channels
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        sigma_layers = [nn.Conv2d(dim, 1, kernel_size=1), ]       # sigma must be positive
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(weights_init)      # xavier init


        self.base_remap_layers = None
        # remap rgb feature dimension
        if self.is_bg:
            out_channels = 2 * out_channels
        base_remap_layers = [nn.Conv2d(dim, out_channels, kernel_size=1), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)

        # using viewdir as condition for rgb feature
        self.rgb_layers = None
        rgb_layers = []
        if self.use_viewdirs:
            dim = out_channels + self.input_ch_viewdirs
            for i in range(1):
                rgb_layers.append(nn.Conv2d(dim, out_channels // 2, kernel_size=1))
                rgb_layers.append(nn.LeakyReLU())
            self.rgb_layers = nn.Sequential(*rgb_layers)


    def forward(self, input, ws):
        '''
        :param input: [..., in_channels+input_ch_viewdirs]
        :ws is a list of size equal to the number of layers
        :return [..., 4]
        '''
        input = input.permute(0,3,1,2)
        input_pts = input[:, :self.in_channel, :, :]
        base = self.base_layers[0]((input_pts, ws[0]))
        for i, (w, layer) in enumerate(zip(ws[1:], self.base_layers[1:])):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=1)
            base = layer((base, w))            
        sigma = self.sigma_layers(base)
        sigma = torch.abs(sigma)

        # rgb feature remap
        base = self.base_remap_layers(base)

        if self.rgb_layers is not None: # adding viewdir condition
            input_viewdirs = input[..., -self.input_ch_viewdirs:, :, :]
            base = self.rgb_layers(torch.cat((base, input_viewdirs), dim=1))
        ret = OrderedDict([('feature', base),
                           ('sigma', sigma.permute(0,2,3,1).squeeze(-1))])
        return ret


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
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


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fg_netdepth = args.fg_netdepth
        self.bg_netdepth = args.bg_netdepth
        self.upsampling_netdepth = args.upsampling_netdepth

        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = Conv1x1Net(w_dim=args.w_dim, D=args.fg_netdepth, out_channels=args.conv_out_channels,
                             in_channels=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = Conv1x1Net(w_dim=args.w_dim, D=args.bg_netdepth, out_channels=args.conv_out_channels // 2,
                             in_channels=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             is_bg=True)

        self.color_mlp = nn.Sequential(
            nn.Conv2d(args.conv_out_channels, args.conv_out_channels*2, kernel_size=1),
            nn.LeakyReLU()
        )
    def forward(self, ws, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        
        ws = ws.unbind(dim=1)
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
        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        
        fg_raw = self.fg_net(input, ws[:self.fg_netdepth])
        fg_raw['feature'] = self.color_mlp(fg_raw['feature']).permute(0,2,3,1)
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
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2,])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input, ws[self.fg_netdepth:self.fg_netdepth+self.bg_netdepth])
        bg_raw['feature'] = self.color_mlp(bg_raw['feature']).permute(0,2,3,1)

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



class SynthesisNetwork(nn.Module):
    def __init__(self,
        args,
    ):
        super().__init__()  
        if isinstance(args, dict):
            from argparse import Namespace
            args = Namespace(**args)
        
        self.cascade_level = args.cascade_level
        self.cascade_samples = [int(x.strip()) for x in args.cascade_samples.split(',')]

        self.models = self.create_nerf(args)

        self.fg_netdepth = args.fg_netdepth
        self.bg_netdepth = args.bg_netdepth
        # Upsampling from composite feature map (fg + bg) 
        is_up = False
        # in_channels = args.conv_out_channels // 2 # not used, for viewdir condiation; conv1x1net output halves the feature dimension
        in_channels = args.conv_out_channels * 2 # start at 32x32x512, double conv_out_channels which is 256
        out_channels = args.up_out_channels
        self.upsampling_layers = []
        for i in range(args.upsampling_netdepth):
            if i % 2 == 0: # upsampling
                is_up = True
                out_channels = in_channels // 2
            else:
                is_up = False
            self.upsampling_layers.append(
                nn.Sequential(
                    SynthesisBLK(w_dim=args.w_dim, in_channels=in_channels, out_channels=out_channels, is_up=is_up),
                    nn.LeakyReLU()
                )                
            )
            in_channels = out_channels

        # to rgb
        self.upsampling_layers.append(
            nn.Sequential(
                SynthesisBLK(w_dim=args.w_dim, in_channels=in_channels, out_channels=3, is_torgb=True),
                nn.Sigmoid()
            )

        )
        self.upsampling_layers = nn.ModuleList(self.upsampling_layers)

        self.H = args.plane_H
        self.W = args.plane_W

        self.camera_intrinsic = get_camera_mat(fov=10)
        self.range_u, self.range_v = [0.24, 0.26], [0.5, 0.5] # control camera postion on a sphere
        self.range_radius = [1, 1]      # scales camera position with sphere radius
        self.N_rand = args.N_rand  
    def create_nerf(self, args):
        models = nn.ModuleDict()
        net = None
        for m in range(args.cascade_level):
            if (args.use_single_nerf) and (net is not None):
                net = net
            else:
                net = NerfNet(args)
            models['net_{}'.format(m)] = net
        return models    

    def forward(self, ws, **kwargs):
        device = ws.device
        batch_size = ws.shape[0]
        pose = get_random_pose(self.range_u, self.range_v, self.range_radius, batch_size=1)
        
        img_raysampler = RaySamplerSingleImage(
            H=self.H,
            W=self.W,
            intrinsics=self.camera_intrinsic,
            c2w=pose.reshape(4,4))        
        # randomly sample rays and move to device
        ray_batch = img_raysampler.random_sample()
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].unsqueeze(0).expand((batch_size,) + ray_batch[key].shape).to(device)
        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        for m in range(self.cascade_level):
            net = self.models['net_{}'.format(m)]

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
        
        ws = ws.unbind(dim=1)
        composite_map = ret['composite_map']
        H = W = int(torch.tensor(composite_map.shape[1]).sqrt().item())
        composite_map = composite_map.reshape(-1, H, W, composite_map.shape[-1]).permute(0,3,1,2)
        for w, up in zip(ws[self.fg_netdepth + self.bg_netdepth: ], self.upsampling_layers):
            composite_map = up((composite_map, w))
        return composite_map

def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5 # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
