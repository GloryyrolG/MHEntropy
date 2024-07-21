import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions
import numpy as np


# DDIM
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# RLE
def fusion(a, b=None, type='cat'):
    if b is None or type is None:
        return a
    if type == 'cat':
        return torch.cat([a, b], dim=-1)
    elif type == 'sum':
        return a + b
    else:  # conv = cat + learnable conv
        raise NotImplementedError


def _reshape_inputs(dims, *args):
    vars = []
    for v in args:
        if isinstance(v, torch.Tensor):
            vars.append(v.reshape(-1, *dims))
        else:  # None
            assert v is None
            vars.append(v)
    return vars


def _tsfm(x, mu=None, logvar=None, reverse=True):
    if reverse:
        # x --> z
        logdet_sigma = torch.zeros(*x.shape[: -1], device=x.device)
        if mu is not None:
            x = x - mu
            if logvar is not None:
                sigma = torch.exp(0.5 * logvar)
                x = x / sigma
                logdet_sigma = -0.5 * torch.sum(logvar, -1)
        return x, logdet_sigma
    else:
        # z --> x
        if mu is not None:
            if logvar is not None:
                sigma = torch.exp(0.5 * logvar)
                x = sigma * x
            x = x + mu
        return x


class _nets(nn.Module):
    def __init__(self, dim, cond_dim=0, h_dims=[64, 64], s=True):
        super(_nets, self).__init__()

        self.cond_dim = cond_dim
        self.s = s
        if self.cond_dim:
            self.fusion_types = ['sum', 'sum']  # current defaults
        else:
            self.fusion_types = [None, None]

        self.l = nn.ModuleList()
        self.l.append(nn.Linear(dim, h_dims[0]))
        self.l.append(nn.Linear(h_dims[0], h_dims[1]))
        self.l.append(nn.Linear(h_dims[1], dim))

        if self.cond_dim:
            self.c = nn.ModuleList()
            for f, h_dim in zip(self.fusion_types, h_dims):
                self.c.append(
                    nn.Linear(self.cond_dim, h_dim) if f else nn.Identity())  # for the None fusion, we also perf a trivial mapping

    def forward(self, x, cond=None):
        """ The conditional way follows CNFs and DDIM (Attn)
        Pattern: (f_x)(x) + f_c(c). Projects into a shared space to fuse
        - NN (sum, sum, None). https://github.com/5yearsKim/Conditional-Normalizing-Flow/blob/master/models/glow/coupling.py#L81
        - ResnetBlock (sum, None). https://github.com/ermongroup/ddim/blob/main/models/diffusion.py#L115
        - ResidualBlock (sum, None). https://github.com/lmnt-com/diffwave/blob/7d1f25582c6ef966eadf49496709f00957cb15e1/src/diffwave/model.py#L106
        - ConcatSquashLinear (affine). https://github.com/luost26/diffusion-point-cloud/blob/910334a8975aa611423a920869807427a6b60efc/models/common.py#L50
        """
        x = self.l[0](x)

        # Can be pre-processed.
        if self.cond_dim > 0:
            conds = [c(cond) for c in self.c]
        else:
            conds = [None] * (len(self.l) - 1)

        for i in range(len(self.l) - 1):
            # 2 cases for no cond: 1. cond=None, 2. No injection (also
            # provide cond, but controlled by fusion_type=None).
            x = fusion(x, conds[i], type=self.fusion_types[i])
            x = F.leaky_relu(x)
            x = self.l[i + 1](x)

        if self.s:
            x = torch.tanh(x)
        return x


class RealNVP(nn.Module):
    """
    https://arxiv.org/pdf/1605.08803.pdf

    Multi-scale arch is not implemented so far
    """
    def __init__(
            self, nets=_nets, nett=_nets, mask=None, prior=None, dim=63,
            tsfm_on=None, kemb=False, jointN=21, h_dims=[64, 64], num_steps=3,
            cond_mapping_dims=None):
        """
        Args:
            dim:
                - 3. Independently models each joint
            tsfm_on (None, 'x', 'z', or int)
                - 'x': modeling a general p(x | I) w/ an actnorm mu | I, sigma | I
                - 'z': modeling p(x | I)
                - int: conditional
            cond_mapping_dims: used when needs to generate conditions for
                different keypoints R^{KC}
        """
        super(RealNVP, self).__init__()

        if dim == 1:
            raise ValueError
        self.dim = dim
        self.jointN = jointN
        if mask is None:
            A = [0] * (dim // 2) + [1] * (dim - dim // 2)
            B = [1 - a for a in A]
            mask = torch.from_numpy(np.array([A, B] * num_steps).astype(np.float32))
        if prior is None:
            prior = distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

        self.tsfm_on = tsfm_on
        cond_dim = self.tsfm_on if type(self.tsfm_on) == int else 0

        # K embs. 2 diff ways.
        if kemb:
            self.ch = 63
            self.kemb_ch = self.ch  # 128 --> 128 * 4 in DDIM
            # timestep embedding
            self.kemb = nn.Sequential(
                torch.nn.Linear(self.ch,
                                self.kemb_ch),
                nn.ReLU(inplace=True),
                torch.nn.Linear(self.kemb_ch,
                                self.kemb_ch),
            )

            if self.kemb_ch != cond_dim:
                cond_dim += self.kemb_ch  # cat
        else:
            # p_k(p_k | z, f). E.g., N(p_k | f_k(z, f), sigma^2I)
            if cond_mapping_dims is None:
                cond_mapping_dims = []
            partitioner = nn.ModuleList()
            for in_features_, out_features_ in cond_mapping_dims:
                assert out_features_ % self.jointN == 0
                partitioner.append(nn.Linear(in_features_, out_features_))
            self.joint_feat_partitioner = partitioner

        self.prior = prior
        self.register_buffer('mask', mask)
        # t and s share similar NNs.
        self.t = torch.nn.ModuleList([
            nett(dim, cond_dim=cond_dim, h_dims=h_dims, s=False)
            for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([
            nets(dim, cond_dim=cond_dim, h_dims=h_dims)
            for _ in range(len(mask))])

        self.scale = 1.  # 1.
        print(f"> Scale {self.scale}x the input")

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z, cond=None):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_, cond=cond) * (1 - self.mask[i])
            t = self.t[i](x_, cond=cond) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x, cond=None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_, cond=cond) * (1 - self.mask[i])
            t = self.t[i](z_, cond=cond) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def make_cond(self, feat):
        """
        1. p | I: F. Joint distribution
        2. p_k | I: joint-independent
            2.1. F_k. Same as p | I
            #TODO: Joint Act Map, Attn
            2.2. F, kemb: follows DDPM

        Args:
            feat: shape: (B, F)
        
        Returns:
            cond: shape: (B, F) or (B * K, C)
        """
        bs = feat.shape[0]
        bs1 = bs * self.jointN if self.dim in [2, 3] else bs

        # 2.2. Add pos embeds.
        if 'kemb' in self._modules:
            assert self.dim in [2, 3]
            # Sharing image features. Repeat #joints times for each image.
            cond = feat[:, None, :].repeat(1, self.jointN, 1).reshape(bs1, -1)

            k = self.kemb(get_timestep_embedding(
                torch.arange(0, self.jointN, device=cond.device).repeat(bs),
                self.ch))
            # Pos embs in Transformers use sum.
            cond = fusion(
                cond, k, type='sum' if cond.shape[1] == k.shape[1] else 'cat')  # simple concat or sum here
        else:  # 1, 2.1
            if self.dim <= 3 and len(self.joint_feat_partitioner):
                joint_feats = []
                p = 0
                for par in self.joint_feat_partitioner:
                    assert len(feat.shape) == 2
                    joint_feats.append(par(feat[:, p: p + par.in_features]).reshape(bs, self.jointN, -1))
                # Ref to https://github.com/luost26/diffusion-point-cloud/blob/910334a8975aa611423a920869807427a6b60efc/models/diffusion.py#L79
                # similar to ours w/ 2 cond inputs.
                feat = joint_feats = torch.cat(joint_feats, dim=-1)  # the simplest concat fusion
            cond = feat.reshape(bs1, -1)
        return cond

    def log_prob(
            self, x, mu=None, logvar=None, return_dict=False, weights=None,
            return_z=False):
        """
        Args:
            x: shape: (B, DK)
            logvar: shape: (B, DK) or (B, F). May also serve as feats
            weights: shape: (B, DK). Visibility. Repeat D times for K joints
        """
        if weights is None:
            weights = torch.ones_like(x)
        
        #TODO: not handle visibility in other dependent modeling cases on COCO.
        if self.dim > 3 and (1 - weights).nonzero().shape[0]:
            raise NotImplementedError

        bs = x.shape[0]
        x, weights = _reshape_inputs([self.dim], x, weights)
        tsfm_on = self.tsfm_on
        if tsfm_on in ['x', 'z']:
            mu, logvar = _reshape_inputs([self.dim], mu, logvar)

        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        logdet_sigma = torch.zeros(x.shape[0], device=x.device)
        if tsfm_on == 'x':
            x, logdet_sigma = _tsfm(x, mu=mu, logvar=logvar)
        else:
            # "Normalize" the input into a reasonable range. No need for
            # tsfm_on == 'x' since we want mu --> x.
            x = x / self.scale

        # Conditional.
        cond = None
        if type(tsfm_on) == int:
            cond = self.make_cond(logvar)
        
        z, logp = self.backward_p(x, cond=cond)

        if tsfm_on == 'z':
            z, logdet_sigma = _tsfm(z, mu=mu, logvar=logvar)

        try:
            loss = ((self.prior.log_prob(z) + logp + logdet_sigma) * weights[:, 0]).view(bs, -1).sum(1)
        except:
            # Hold on for debugging.
            pass
        if return_z:
            return z, loss
        elif return_dict:
            return {
                'loss': loss,
            }
        else:
            return loss

    def sample(self, batchSize, temp=0.7, mu=None, logvar=None, return_z=False):
        bs = mu.shape[0] if mu is not None else logvar.shape[0]
        tsfm_on = self.tsfm_on
        if tsfm_on in ['x', 'z']:
            mu, logvar = _reshape_inputs([self.dim], mu, logvar)

        z0 = z = self.prior.sample((batchSize,)).cuda() * temp

        if tsfm_on == 'z':
            z = _tsfm(z, mu=mu, logvar=logvar, reverse=False)

        cond = None
        if type(tsfm_on) == int:
            cond = self.make_cond(logvar)

        x = self.forward_p(z, cond=cond)

        if tsfm_on == 'x':
            x = _tsfm(x, mu=mu, logvar=logvar, reverse=False)
        else:
            # "Inv-normalize" back to the original scale.
            x = x * self.scale
        
        if return_z:
            return x.view(bs, -1), z0.view(bs, -1)
        else:
            return x.view(bs, -1)

    def forward(self, x):
        return self.log_prob(x)


def combine_flow_cond(**kwargs):
    """
    Args:
        f: image feats
    """
    return torch.cat([kwargs['z'], kwargs['f']], dim=1)
