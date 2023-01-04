import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
import os
from functools import lru_cache

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
    'morton': 2,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners

        return outputs

    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None



grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear'):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation] # "linear" or "smoothstep"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)

        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.

        D = self.input_dim
        C = self.embeddings.shape[1] # embedding dim for each level
        L = self.offsets.shape[0] - 1 # level
        S = np.log2(self.per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')

        _backend.grad_total_variation(inputs, self.embeddings, self.embeddings.grad, self.offsets, weight, B, D, C, L, S, H, self.gridtype_id, self.align_corners)

    class MultiSceneGridEncoder(nn.Module):
        count = 0 # global counter for unique name
        """
        Multi-scene grid encoder, each scene has its own grid encoder.
        As storing all the grid encoders in GPU memory is impractical, they are stored on disk and loaded on demand.
        """
        def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear', workdir='encoder', device='cuda', max_encoders_in_device=8):
            super().__init__()
            self.encoder_args = dict(
                input_dim=input_dim,
                num_levels=num_levels,
                level_dim=level_dim,
                per_level_scale=per_level_scale,
                base_resolution=base_resolution,
                log2_hashmap_size=log2_hashmap_size,
                desired_resolution=desired_resolution,
                gridtype=gridtype,
                align_corners=align_corners,
                interpolation=interpolation,
            )
            self.scene_encoders = {}
            self.latest_used = {}
            # get unique name
            self.encoder_name = f'encoder_{type(self).count}'
            self.workdir = workdir
            self.device = device
            self.usage_counter = 0
            self.max_encoders_in_device = max_encoders_in_device

        def fetch_scene_encoders(self, scene_indices):
            # scene_indices: list/Tensor of scene indices
            # fetch scene encoders from disk and store in GPU memory, and register them as submodules
            for scene_index in scene_indices:
                if scene_index not in self.scene_encoders:
                    self.scene_encoders[scene_index] = GridEncoder(**self.encoder_args)
                elif isinstance(self.scene_encoders[scene_index], str):
                    # load from disk
                    encoder_path = os.path.join(self.workdir, self.scene_encoders[scene_index])
                    self.scene_encoders[scene_index] = GridEncoder(**self.encoder_args)
                    self.scene_encoders[scene_index].load_state_dict(torch.load(encoder_path))
                self.scene_encoders[scene_index].to(self.device)
                self.register_module(f'{self.encoder_name}_{scene_index}', self.scene_encoders[scene_index])
                self.latest_used[scene_index] = self.usage_counter
                self.usage_counter += 1
        def remove_cold_encoders_from_device(self):
            if len(self.scene_encoders) > self.max_encoders_in_device:
                # remove cold encoders from device
                cold_encoders = sorted(self.latest_used.items(), key=lambda x: x[1])[:len(self.scene_encoders) - self.max_encoders_in_device]
                for scene_index, _ in cold_encoders:
                    # move to hard disk
                    encoder_path = os.path.join(self.workdir, f'{self.encoder_name}_{scene_index}.pt')
                    torch.save(self.scene_encoders[scene_index].state_dict(), encoder_path)
                    self.scene_encoders[scene_index] = encoder_path
                    self.latest_used.pop(scene_index)

        def get_active_parameters(self):
            """
            it seems that pytorch does not support removing submodules, so we need to prvoide such a function
            to manually provide the parameters to be optimized to the optimizer.
            """
            active_parameters = []
            for scene_index, encoder in self.scene_encoders.items():
                if not isinstance(encoder, str):
                    active_parameters += list(encoder.parameters())
            return active_parameters


        def forward(self, inputs, scene_indices, bound=1):
            # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
            # scene_indices: [...], scene index for each input
            # return: [..., num_levels * level_dim]
            outputs = []
            for scene_id in scene_indices:
                assert scene_id in self.scene_encoders, f'encoder for scene {scene_id} not found!'
                assert not isinstance(self.scene_encoders[scene_id], str), f'encoder for scene {scene_id} not loaded!'
                outputs.append(self.scene_encoders[scene_id](inputs[scene_id].unsqueeze(0), bound=bound))
            return torch.cat(outputs, dim=0)



