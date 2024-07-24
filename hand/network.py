import torch
from torch import nn
import torch.distributions.normal
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

from ManoLayer import ManoLayer
from criteria import chamfer_dist
from utils import batch_normalize_pose3d
POINT_SIZE = 256
DecPointSize = 256
DecUVSize = np.sqrt(DecPointSize)
import torch
from torch import nn
from torch import distributions as D
import numpy as np
from typing import Union, Optional
from flows import RealNVP
from nflows.flows import ConditionalGlow

import trimesh
from viz import viz_2djoints, mesh_axis_tsfm
import time


class BasicEnc(nn.Module):
    """ A simple multi-head net

    Attributes:
        l: connector
        l1: coordinate branch

    Args:
        cfg: a placeholder for class inheritance
        n_latent: also used to repr the output
        feat_dim: for FC head input. Sometimes needs to calc

    Returns:
        - mu (e.g., R^{64}), logvar
        - discriminative pred (e.g., R^{63}), feat (e.g., sigma)
        - repr_1, repr_2 (e.g., SimCC)
    """
    def __init__(
            self, cfg=None, n_latent: Union[int, list] = 64, backbone='resnet18',
            pretrained=True, conditional_p=False, K=21, D=3, feat_dim=None,
            sigma_act='exp', deterministic=False, **kwargs):
        super(BasicEnc, self).__init__()

        if type(n_latent) == int:
            self.n_latent = [n_latent, n_latent]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if backbone == 'resnet18':
            self.res = models.resnet18(pretrained=pretrained).to(self.device)
        elif backbone == 'resnet50':
            self.res = models.resnet50(pretrained=pretrained).to(self.device)
        else:
            raise NotImplementedError
        # Simplify.
        self.res.fc = nn.Identity()

        #TODO: deprecate conditional_p.
        in_dim = 1000
        self.conditional_p = conditional_p
        if self.conditional_p:
            in_dim += K * D

        # Connectors.
        self.preavgpool = None  # placeholder impl
        self.l = nn.Identity()
        
        if backbone == 'poseresnet18':
            self.l1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(start_dim=1))
            self.l2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(start_dim=1))
        else:
            if feat_dim is None:
                # feat_dim does not change.
                feat_dim = {
                    'resnet18': 512,
                    'resnet50': 2048
                }[backbone]
            self.l1 = nn.Sequential(nn.Linear(feat_dim, self.n_latent[0]))
            self.l2 = nn.Sequential(nn.Linear(feat_dim, self.n_latent[1]))

        self.sigma_act = sigma_act
        self.deterministic = deterministic  # for AEs

        # Store the feat during FF.
        self._feat = None

    def forward(self, x, deterministic=False, p=None):
        """
        (To remove)
        - VAEs:
          - (Current) p(p | I) = \int_z p(p | z) p(z | I) dz;
            Encoder: q(z | I)?

          - (Conditional, very rich info) p(p | I) = \int_z p(p | I, z) p(z | I) dz,
            where p(z | I) can be N(0, I);
            Encoder: q(z | I, p) or q(z | I). Maybe we can use F to replace I.
            It is reconstructing p.
        
        - Cross-Modal VAE: Ez∼q(z|xi)[log p(xt|z)] − DKL(q(z|xi)||p(z))
        """
        x=self.res(x)  # (B, 512, 8, 8) bef avg
        if self.conditional_p:
            x = torch.cat([x, p.flatten(start_dim=1)], dim=1)

        # Use pre-avgpool feats.
        if self.preavgpool:
            x = self.preavgpool.features_input  # shape: 4D

        self._feat = x        
        x = self.l(x)

        mn = self.l1(x)
        sd = self.l2(x)
        if self.sigma_act == 'exp':
            # l2: logvar.
            sd = torch.exp(0.5 * sd)
        elif self.sigma_act == 'sigmoid':
            # (0, 1).
            sd = torch.sigmoid(sd)

        m = torch.distributions.normal.Normal(torch.zeros_like(mn), torch.ones_like(mn))
        epsilon = m.sample()
        if (self.deterministic or deterministic
                or mn.shape != sd.shape):
            z = mn
        else:
            z = mn + sd * epsilon

        # z = mn = sd = self.gap(F.relu(self.depthconv(self.fm64.features_output), inplace=True)).flatten(start_dim=1)

        return z,mn,sd


class _ApproxUniform(nn.Module):
    """ \log\tilde{p}(x) (non-batch for rec, batch for ball) """
    def __init__(self, a, b, alpha=1., sup='rec'):
        super(_ApproxUniform, self).__init__()
        self.a = a  # shape: (*, D)
        self.b = b  # shape: (*,)
        self.alpha = alpha
        self.sup = sup

    def forward(self, x):
        pass

    def log_prob(self, x):
        if self.sup == 'rec':
            return -(self.alpha * F.relu(
                (x - (self.a + self.b) / 2.).abs() / ((self.b - self.a) / 2.) - 1.) ** 2).sum(1)  # actually is \tilde{p}
        elif self.sup == 'ball':
            r = (x - self.a).norm(p=2, dim=-1)
            assert (isinstance(self.b, torch.Tensor) and torch.all(self.b > 0.)  # currently only supports (batch-wise) scalars
                    or self.b > 0.)
            return -self.alpha * F.relu(r / self.b - 1.) ** 2
        else:
            raise NotImplementedError

    def sample(self, sample_shape=torch.Size([]), device='cuda'):
        # Arg check.
        if type(sample_shape) is int:
            sample_shape = torch.Size((sample_shape,))
        if self.sup == 'rec':
            return torch.rand(*sample_shape, device=device) * (self.b - self.a) + self.a
        elif self.sup == 'ball':
            # https://stackoverflow.com/a/50746409
            r = self.b * (torch.rand(*sample_shape, device=device)) ** 0.5
            x = torch.randn(*sample_shape, self.a.shape[-1], device=device)
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-16) * r[..., None]
            return x + self.a
        else:
            raise NotImplementedError


class _LogDist(nn.Module):
    def __init__(self, dist='normal', **dist_kwargs):
        super(_LogDist, self).__init__()
        if dist == 'normal':
            # Requires the support > 0!
            self.dist = D.Normal(dist_kwargs['loc'], dist_kwargs['scale'])
        else:
            raise NotImplementedError

    def forward(self, x):
        pass

    def log_prob(self, x):
        """ \log p_{\log s}(\log s)=\log p_s(s)-\log s """
        return self.dist.log_prob(x) - torch.log(x)


class _Laplace(nn.Module):
    """ Cond

    Attrs:
        kwargs: - in_dim, out_dim, dist, alpha
    """
    def __init__(self,
            b_type='const', b_init=0.05, **kwargs):
        super(_Laplace, self).__init__()
        self.b_type = b_type
        if self.b_type == 'const':
            self.register_buffer('b', torch.ones(1, device='cuda') * b_init)  # 2. / 42 / 2. = 0.02
            # For ApproxUniform hybrid.
        elif self.b_type == 'scalar':
            self.register_parameter('b', nn.Parameter(torch.ones(1, device='cuda') * b_init))
        elif self.b_type == 'nn_diag':  # diag, dependent
            self.b = nn.Sequential(
                nn.Linear(kwargs['in_dim'], kwargs['out_dim']),
            )  # log_b or \log\var
        else:
            raise NotImplementedError
        print(f"> b {b_init:.4f}")
        if 'dist' in kwargs:
            self.dist = kwargs['dist']
        else:
            self.dist = D.Laplace(
                torch.zeros([], device='cuda'), torch.ones([], device='cuda'))
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
    
    def forward(self, x):
        pass

    def log_prob(self, x, mu, weights=None, feat=None, **kwargs):
        """
        Args:
            patch
        """
        log_p = 0.
        if self.b_type == 'nn_diag':
            b = self.b(feat).exp()
        else:
            b = self.b
        if weights is None:
            weights = torch.ones_like(mu)
        else:
            pass

        # Not add on root_idx.
        if isinstance(self.dist, RealNVP):
            log_p = log_p + self.dist.log_prob(x, mu=mu + 1e-4, logvar=torch.log(b), weights=weights == 1)
            # Using residual.
            b = torch.tensor([1], dtype=torch.float32, device='cuda')
            # return log_p
        # 2.8e-4 (uv); 1.9e-3 (xyz)
        log_p = log_p + (
            (weights == 1.)
            * (-(F.relu((x - mu).abs() - 1e-4) + 1e-4) / b - torch.log(2 * b))).flatten(start_dim=1).sum(1)
        return log_p

    def sample(self, mu, feat=None):
        if self.b_type == 'nn_diag':
            b = self.b(feat).exp()  # shape: (NB, KD)
        else:
            b = self.b
        e = self.dist.sample(sample_shape=mu.shape)
        return mu + b * e


class _Categorical(nn.Module):
    def __init__(self, in_channels, K, hdims):
        super(_Categorical, self).__init__()
        self.nn = [
            nn.Linear(in_channels, hdims[0]),
            nn.ReLU(inplace=True)]
        for l in range(1, len(hdims)):
            self.nn.extend([
                nn.Linear(hdims[l - 1], hdims[l]),
                nn.ReLU(inplace=True)])
        self.nn.append(nn.Linear(hdims[-1], K))
        self.nn = nn.Sequential(*self.nn)

    def forward(self, x):
        raise NotImplementedError

    def log_prob(self, x=None, cond=None):
        if x is not None:
            raise NotImplementedError
        else:
            return F.log_softmax(self.nn(cond), dim=-1)

    def sample(self, cond, temp=1):
        """ Usually temp is not used """
        probs = F.softmax(self.nn(cond) / (temp + 1e-16), dim=-1)
        dist = D.Categorical(probs=probs)  # supports batch
        return dist.sample()


class _DummyDist(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_DummyDist, self).__init__()

    def log_prob(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs):
        return


class MHEnt(nn.Module):
    r""" PGM: I --> z = \theta, \beta, s, t --> y

    Attrs:
        image_size: scalar

    Args:
        special_cfg:
            - q_z_giv_i_model
            - q_z_giv_i_cfg
            - ds
            - image_size
            - mano_cfg
                - flat_hand_mean
                - ncomps
                - use_pca
            - prior_cfg
                - p_theta45_pth
                - th45_ref_alpha
            - kld_w, T
    """
    def __init__(self, special_cfg, **common_cfg):
        super(MHEnt, self).__init__()

        self.integrated = True  # a integrated model w/ a feat extractor, Enc, and Dec
        if common_cfg['input'] == 'image':
            self.feat_extractor = BasicEnc(**common_cfg)
        else:
            raise NotImplementedError
        # p(\theta,\beta,s,t|f(I)) or q.
        q_z_giv_i_model = special_cfg['q_z_giv_i_model']
        if q_z_giv_i_model == 'realnvp':
            self.q_z_giv_i = RealNVP(**special_cfg['q_z_giv_i_cfg'])
        elif q_z_giv_i_model == 'glow':
            self.q_z_giv_i = ConditionalGlow(
                45, 512, 4, 2, context_features=512, dropout_probability=0.2)
        elif q_z_giv_i_model == 'det':
            self.q_z_giv_i = _DummyDist()
        else:
            # raise NotImplementedError
            self.q_z_giv_i = _DummyDist()  # requires override
        self.surrogate_3d = None
        if self.surrogate_3d:
            self.q_z_giv_z = _Laplace(b_init=0.01)  # together w/ reweighting
        
        self.ds = special_cfg['ds']
        self.image_size = max(special_cfg['image_size'])  # scalar
        mano_cfg = special_cfg['mano_cfg']
        flat_hand_mean = mano_cfg['flat_hand_mean']
        ncomps = mano_cfg['ncomps']
        use_pca = mano_cfg['use_pca']
        self.mano_dec = ManoLayer(
            skeidx='RHD',  # notice! Alias
            flat_hand_mean=flat_hand_mean, ncomps=ncomps, use_pca=use_pca,
            output_size=self.image_size, mask_sz=64)  # a plain MANO Dec

        # Z config. Currently, supports 3 types of q(z|I): prob, (almost) det,
        # and GT data.
        self.zdims = {
            'th3': 3, 'th45': 45, 'bt': 10, 'logs': 1, 't': 2
        }
        self.zdets = {
            'th3': True, 'th45': q_z_giv_i_model == 'det', 'bt': True, 'logs': True, 't': True
        }
        self.z_dim = sum([v for k, v in self.zdims.items() if not self.zdets[k]])

        # q(z|I)
        feat_dim = 512  # self.feat_extractor.n_latent[0]
        det_head_hdim = feat_dim
        det_head_odim = sum([v for k, v in self.zdims.items() if self.zdets[k]])
        if det_head_odim:
            self.det_head = nn.Sequential(
                nn.Linear(feat_dim, det_head_hdim),
                nn.ReLU(inplace=True),
                nn.Linear(det_head_hdim, det_head_odim))
        else:
            self.det_head = nn.Identity()
        
        self.loss_weights = {}

        ######### Reconsts. st is modeled together w/ \theta and \beta.
        data_prior_cfg = special_cfg['data_prior_cfg']
        self.p_y_giv_mus = {}
        self.p_y_giv_mus['uv'] = _Laplace(b_type='const', b_init=data_prior_cfg['b_2d'], **{'alpha': 5.})  # 'scalar')  # th_bt_logs_t
        self.p_y_giv_mus['xyz'] = _Laplace(b_type='const', b_init=0.03)  # under tuning
        # self.p_m_giv_mu = _Laplace(b_type='const', b_init=50.)  # 0.1
        learnable_reconst = False  # True
        if not learnable_reconst:
            # Const.
            self.p_y_giv_ys = {}
            self.p_y_giv_ys['uv'] = self.p_y_giv_mus['uv']
            self.p_y_giv_ys['xyz'] = self.p_y_giv_mus['xyz']
        else:
            # Learnable.
            self.p_y_giv_ys = nn.ModuleDict()
            self.p_y_giv_ys['uv'] = _Laplace(
                b_type='nn_diag',
                **{'alpha': 5., 'in_dim': feat_dim, 'out_dim': common_cfg['K'] * 2})
                #    'dist': RealNVP(dim=2, tsfm_on='x', h_dims=[256, 256], num_steps=6)})  # RLE
            self.p_y_giv_ys['xyz'] = _Laplace(
                b_type='nn_diag',
                **{'in_dim': feat_dim, 'out_dim': common_cfg['K'] * 3})

        self.p_ys = {}

        #########

        ######### Priors.
        self.ps = {}
        # p(\theta[3:]). Additionally, p(s) is Uniform * Normal w/o any prior?
        prior_cfg = special_cfg['prior_cfg']
        p_theta45_pth = prior_cfg.get('p_theta45_pth', None)
        if p_theta45_pth:
            print(f"> Loadin {p_theta45_pth}")
            self.ps['th45'] = self.p_th45 = VAE4Pose(
                latent_dim=32, model_path=p_theta45_pth)  # consistent scaling direction using ELBO
            self.ps['th45'].requires_grad_(False)  # frozen params
        # Tilde energy.
        th45_ref_alpha = prior_cfg.get('th45_ref_alpha', 50.)
        if use_pca:
            self.ps['th45_ref'] = _ApproxUniform(-2., 2., alpha=th45_ref_alpha)  # use the official PCA prior
        else:
            self.ps['th45_ref'] = _ApproxUniform(
                torch.zeros(45, device='cuda'), np.pi, alpha=th45_ref_alpha, sup='ball')  # ref/base dist. Tuning the penalty
        self.ps['th3_ref'] = _ApproxUniform(
            torch.zeros(3, device='cuda'), np.pi, alpha=5., sup='ball')
        self.ps['bt'] = _ApproxUniform(-0.03, 0.03, alpha=50.)  # initially not in the range

        #########

        self.kld_w = self.kld_w_final = special_cfg.get('kld_w', 1.)
        self.kld_w_annealing = special_cfg['kld_w_annealing']
        self.T = special_cfg.get('T', 1.)
        self.entropy = special_cfg['loss_cfg']['entropy']  # options: True, False, 'infogan'

        self.best_mode = special_cfg['loss_cfg']['mode']  # False

        ######### Data bias prior.
        if self.best_mode:
            pass

        #########

        self._extra_ws = {}
        self._extra_ws['log_p_vis_giv_z'] = 1.

    def _th_bt_dec(self, th_bt) -> dict:
        """
        Args:
            th_bt: shape: (NB, Tb) or (N, B, Tb)
        """
        if len(th_bt.shape) == 3:
            th_bt = th_bt.reshape(-1, th_bt.shape[-1])
        theta, beta = th_bt[:, : 48], th_bt[:, -10:]
        out = self.mano_dec(beta=beta, theta=theta)  # shape: (NB, K, D)
        return out

    def _choose_xyz_from_dec(self, out: dict, return_verts=False):
        """
        Returns:
            xyz: shape: (NB, K, D). Normed rel
            verts (Optional): normed rel
        """
        if self.ds in ['rhd', 'ho3d']:
            xyz = out['mano_joints']  # mm
        else:
            raise NotImplementedError
        normed_rel_xyz, t, s = batch_normalize_pose3d(
            xyz, {'rhd': 12, 'freihand': 9, 'ho3d': 12}[self.ds],
            norm_idx={'rhd': 11, 'freihand': 10, 'ho3d': 11}[self.ds], return_st=True)
        if return_verts:
            verts = (out['mesh'] - t) / s[:, None, None]
            return normed_rel_xyz, verts, s
        else:
            return normed_rel_xyz
    
    def _th_bt2xyz(self, th_bt, return_verts=False):
        """
        Args:
            th_bt: shape: (NB, Tb) or (N, B, Tb)
        
        Returns:
            xyz: shape: (NB, K, D). Normed rel
        """
        out = self._th_bt_dec(th_bt)
        xyz = self._choose_xyz_from_dec(out, return_verts=return_verts)
        return xyz 
    
    def _orth_proj(self, xyz, logs_t, inv_norm=False):
        """
        Args:
            xyz: shape: (NB, K, 3) or (N, B, K, 3)
            logs_t: shape: (NB, 3) or (N, B, 3). Camera projection
        
        Returns:
            uv: shape: (NB, K, 2)
        """
        s_t = logs_t.clone()
        s_t[..., 0] = torch.exp(s_t[..., 0])  # scale > 0.
        if len(xyz.shape) == 4:
            xyz = xyz.flatten(0, 1)
        if len(s_t.shape) == 3:
            s_t = s_t.flatten(0, 1)
        uv = self.mano_dec.batch_orth_proj(
            xyz, s_t[:, [0]], s_t[:, -2:], self.image_size, inv_norm=inv_norm)
        return uv

    def _th_bt2uv(self, th_bt, logs_t):
        """
        Args:
            th_bt: shape: (NB, Tb) or (N, B, Tb)
            logs_t: shape: (NB, 3) or (N, B, 3)
        
        Returns:
            uv: shape: (NB, K, D)
        """
        xyz = self._th_bt2xyz(th_bt)
        return self._orth_proj(xyz, logs_t)

    def _v2render(self, verts, norm, logs_t, render=['mask']) -> dict:
        assert len(verts.shape) == 3 and len(norm.shape) == 1
        if len(logs_t.shape) == 3:
            logs_t = logs_t.flatten(0, 1)
        render_out = self.mano_dec.render(
            logs_t[:, [0]].exp(), logs_t[:, -2:], vertex=verts, norm=norm,
            render=render)
        return render_out

    def _th_bt2render(self, th_bt, logs_t, render=['mask']) -> dict:
        _, verts, norm = self._th_bt2xyz(th_bt, return_verts=True)
        return self._v2render(verts, norm, logs_t, render=render)

    def _th_bt_product(self, th_bt, logs_t, mods=None, inv_norm=False) -> dict:
        """
        Returns:
            - uv: shape: (NB, K, D)
            - mask, depth: shape: (NB, 64, 64)
        """
        output = {}
        dec_d = self._th_bt_dec(th_bt)
        output['xyz'], output['verts'], norm = self._choose_xyz_from_dec(dec_d, return_verts=True)
        if 'uv' in mods:
            output['uv'] = self._orth_proj(output['xyz'], logs_t, inv_norm=inv_norm)
        render = []
        if 'm' in mods:
            render.append('mask')
        if 'depth' in mods:
            render.append('depth')
        output.update(self._v2render(output['verts'], norm, logs_t, render=render))
        return output
    
    @classmethod
    def _set_evidences_(cls, z, evidences: Optional[dict] = None):
        """ In-place. """
        # Use GT.
        if evidences is None:
            return
        if 'bt' in evidences:
            z.data[:, 48: 58] = evidences['bt']
        if 'logs' in evidences:
            z.data[:, [-3]] = evidences['logs']
            # z.data[:, -3] = np.log(0.31)  # logs
        if 't' in evidences:
            z.data[:, -2:] = evidences['t']

    def _sample_p_z(self, sample_shape=torch.Size([]), **kwargs):
        output = {}
        if len(sample_shape) == 1:
            N, bs = 1, sample_shape[0]
        else:
            N, bs = sample_shape
        for param_s, dim in self.zdims.items():
            sampler = None
            if f'{param_s}_mean' in kwargs:
                mean = kwargs[f'{param_s}_mean']
                std = mean.std(0)
                output[param_s] = mean + torch.randn_like(mean) * std * 0.3
            elif f'{param_s}_ref' in self.ps:
                sampler = self.ps[f'{param_s}_ref']
            elif param_s in self.ps:
                sampler = self.ps[param_s]
            else:
                sampler = D.Normal(
                    torch.zeros([], device='cuda'), torch.ones([], device='cuda'))
            if sampler:
                if len(getattr(sampler, 'event_shape', torch.Size([]))):
                    sample_shape_t = (N * bs,)
                else:
                    sample_shape_t = (N * bs, dim)
                output[param_s] = sampler.sample(sample_shape_t)
        return torch.cat(tuple(output.values()), dim=1)

    def _sample_p_d(self, y, use_gt: list, N=1) -> dict:
        evidences = {}
        if 'bt' in use_gt:
            evidences['bt'] = torch.zeros(N * y['st'].shape[0], 10, device='cuda')
        st_evi = y['st'].repeat(N, 1)  # double --> float
        if 'logs' in use_gt:
            evidences['logs'] = st_evi[:, [0]].log()
        if 't' in use_gt:
            evidences['t'] = st_evi[:, -2:]
        return evidences

    def _forward_log_p(
            self, z, y, use_gt=[], mods=None, feat=None, return_dict=True, **kwargs) -> dict:
        r""" -\log p(uv|z)\tilde{p}(z). If uses GT \log s and t, then ancestral
        sampling w/ p(z)=p_d(\log s,t\|I)

        Args:
            kwargs: vis_w
        """
        if mods is None:
            mods = ['uv']  # , 'xyz']
        output = {}
        th_bt = z[:, : 58]  # shape: (B, 58)
        logs_t = z[:, -3:]
        dec_out = self._th_bt_product(th_bt, logs_t, mods=mods)
        N = z.shape[0] // y['crop_uv'].shape[0]
        for mod in ['uv', 'xyz']:
            if mod not in mods:
                continue
            mu = dec_out[mod].flatten(start_dim=-2)
            # \log\E_p[p]=\log\mean\exp\log p\approx\log p
            # \geq\E_p[\log p].
            # 1. Analytical sol to Laplaces;
            # 2. Matching by sampling.
            _y = mu  # self.p_y_giv_mus[mod].sample(mu, feat=feat)
            y_ = y[{'uv': 'crop_uv', 'xyz': 'pose3d'}[mod]]
            D = {'uv': 2, 'xyz': 3}[mod]
            # weights = None
            if kwargs.get('vis_w', True):
                weights = y['vis'][..., None].repeat(N, 1, D).flatten(start_dim=-2) 
            output[f'log_p_{mod}_giv_z'] = self.p_y_giv_ys[mod].log_prob(
                y_.repeat(N, 1), _y, feat=feat, weights=weights,)
                # **{'patch': y['patch'].repeat(N, 1)})
            
        # q(\theta,\beta,s,t|I)\log p(\theta,\beta,s,t).
        th3, th45, bt, logs, t = z[:, : 3], z[:, 3: 48], z[:, 48: 58], z[:, [-3]], z[:, -2:]
        for param_s in self.zdims:
            if param_s in use_gt:
                continue
            output[f'log_p_{param_s}'] = 0.
            if f'{param_s}_ref' in self.ps:
                output[f'log_p_{param_s}'] = (
                    output[f'log_p_{param_s}']
                    + eval(f'self.ps["{param_s}_ref"].log_prob({param_s})'))
            if param_s in self.ps:
                if param_s == 'th45':
                    self.ps['th45'].requires_grad_(False)  # frozen params
                    self.ps['th45'].eval()  # BNs and Dropouts
                output[f'log_p_{param_s}'] = (
                    output[f'log_p_{param_s}']
                    + eval(f'self.ps["{param_s}"].log_prob({param_s})'))
        output['log_p'] = sum(output.values()) / self.T

        if return_dict:
            return output
        else:
            return output['log_p']

    def _reverse_log_q(self, z, feat, return_dict=False, **kwargs) -> dict:
        """
        Args:
            z: shape: (NB, Z)
            feat: shape: (NB, F)
            kwargs: - y, zz
        """
        output = {}
        output['log_q'] = 0.

        full_z = z

        z, z_det = [], []
        p_t = 0
        for param_s, ndim in self.zdims.items():
            if self.zdets[param_s]:
                z_det.append(full_z[:, p_t: p_t + ndim])
            else:
                z.append(full_z[:, p_t: p_t + ndim])
            p_t = p_t + ndim
        if len(z):
            z = torch.cat(z, dim=1)
            if isinstance(self.q_z_giv_i, RealNVP):
                output['log_q_z_giv_i'] = self.q_z_giv_i.log_prob(z, logvar=feat)
            elif isinstance(self.q_z_giv_i, ConditionalGlow):
                output['log_q_z_giv_i'], _ = self.q_z_giv_i.log_prob(z, context=feat)
            output['log_q'] = output['log_q'] + output['log_q_z_giv_i']
        output['log_q_zd_giv_i'] = 0.  # self.q_zd_giv_i.log_prob(z_det)  #TODO: add st distribution loss
        output['log_q'] = output['log_q'] + output['log_q_zd_giv_i']
        if not return_dict:
            return output['log_q']
        else:
            raise NotImplementedError
    
    def _combine_z(self, z_det, z=None, p_d={}, use_gt=[]):
        full_z = []
        # Use 2 pointers instead of the dict.
        p_t, q_t = 0, 0
        for param_s, ndim in self.zdims.items():
            if param_s in use_gt:
                full_z.append(p_d[param_s])
            elif self.zdets[param_s]:
                full_z.append(z_det[:, q_t: q_t + ndim])
                q_t = q_t + ndim
            else:
                full_z.append(z[:, p_t: p_t + ndim])
                p_t = p_t + ndim
        full_z = torch.cat(full_z, dim=1)
        return full_z

    def _sample_q_z_giv_i(
            self, feat, N=1, temp=1., return_dict=False, y=None, **kwargs):
        """
        Args:
            feat: shape: (B, F)
            kwargs: return_zz, return_log_p

        Returns:
            z: shape: (NB, Z)
        """
        # Use GTs (debug).
        use_gt = []  # ['bt', 'logs', 't']
        p_d = self._sample_p_d(y, use_gt, N=N)
        if isinstance(self.q_z_giv_i, RealNVP):
            out = z = self.q_z_giv_i.sample(
                N * feat.shape[0], temp=temp, logvar=feat.repeat(N, 1),
                return_z='return_zz' in kwargs)
        elif isinstance(self.q_z_giv_i, ConditionalGlow):
            # z, log_prob_t, z_t = self.q_z_giv_i.sample_and_log_prob(N, noise=None, context=feat)
            noise = torch.randn(feat.shape[0], N, *self.q_z_giv_i._distribution._shape,
                                dtype=torch.float32, device='cuda') * temp  # N(0, 1) base
            z, log_prob, z_t = self.q_z_giv_i.sample_and_log_prob(N, noise=noise, context=feat)
            z = z.permute((1, 0, 2)).flatten(0, 1)
            log_prob = log_prob.transpose(0, 1).flatten()
        else:
            z = None
        if 'return_zz' in kwargs:
            z, zz = out
        z_det = self.det_head(feat).repeat(N, 1)
        # self.q_zd_giv_i.loc(feat).repeat(N, 1)  # use mean
        full_z = self._combine_z(z_det, z=z, p_d=p_d, use_gt=use_gt)
        if not return_dict:
            if 'return_zz' in kwargs:
                return full_z, zz
            elif 'return_log_p' in kwargs:
                return full_z, log_prob
            else:
                return full_z
        else:
            raise NotImplementedError

    def _reverse_kld(
            self, y: dict, x, mods=None, return_dict=True
            ) -> Union[dict, torch.Tensor]:
        r""" # \log p(y|I)>=E_{p(\theta,\beta,s,t|I)}[\log p(y|\theta,\beta,s,t)]

        -KL=E_{q(\theta,\beta,s,t|I)}[\log p(y|\theta,\beta,s,t)]
        -KL(q(\theta,\beta,s,t|I)\|p(\theta,\beta,s,t))

        #TODO: consider vis

        Returns:
            output: for criteria
                - log_p or neg_kl
                - th_norm, bt_norm
        """
        if mods is None:
            mods = ['uv']
        output = {}
        if isinstance(self.feat_extractor, BasicEnc):
            _, feat, _1 = self.feat_extractor(x)
        N = 10  # converged faster than 1
        if isinstance(self.q_z_giv_i, ConditionalGlow):
            # Since uses Dropout.
            z, log_q_z_giv_i = self._sample_q_z_giv_i(feat, N=N, y=y, **{'return_log_p': True})
        else:
            z = self._sample_q_z_giv_i(feat, N=N, y=y)
        th_bt = z[:, : 58]  # shape: (B, 58)
        output['th_norm'] = th_bt[:, : 48].norm(p=2, dim=1)
        output['bt_norm'] = th_bt[:, -10:].norm(p=2, dim=1)

        output['log_p'] = 0.
        q_log_p_d = self._forward_log_p(z, y, use_gt=[], mods=mods, feat=feat)  # use_gt includes bt, logs, t
        # Reconst.
        output['q_log_p_z_giv_y'] = q_log_p_d['log_p'].reshape(N, -1).mean(0)
        
        # Entropy.
        if self.entropy:
            kwargs = {}
            if isinstance(self.q_z_giv_i, ConditionalGlow):
                h_q_z_giv_i = -log_q_z_giv_i
            else:
                h_q_z_giv_i = -self._reverse_log_q(z, feat.repeat(N, 1), **kwargs)
            output['h_q_z_giv_i'] = h_q_z_giv_i.reshape(N, -1).mean(0)
            output['log_p'] = output['log_p'] + output['h_q_z_giv_i']

        # ELBO.
        output['log_p'] = (output['log_p'] + 
                           # output['q_log_p_y_giv_z'] - self.kld_w * output['kl_q_z_giv_i_w_p_z'])  # deprecated
                           output['q_log_p_z_giv_y'])

        if self.p_ys.get('uv', None) is not None:
            uv_wo_rot = ((y['crop_uv'] + 1) * 128).reshape(*y['crop_uv'].shape[: -1], -1, 2)
            # uv_wo_rot = torch.cat([uv_wo_rot,
            #                        torch.ones(*uv_wo_rot.shape[: -1], 1, device='cuda')], dim=-1)
            # uv_wo_rot = uv_wo_rot.matmul(y['rot_mat_inv'])
            uv_wo_rot = uv_wo_rot / 256 * 2 - 1
            uv_wo_rot = uv_wo_rot - uv_wo_rot[:, [12]]
            uv_wo_rot = uv_wo_rot.flatten(-2)
            output['log_p_uv_real'] = self.p_ys['uv'].log_prob(uv_wo_rot + torch.randn_like(uv_wo_rot) * 1e-4)
            output['log_p'] = output['log_p'] + output['log_p_uv_real']

        use_chamfer_loss = False  # False
        if use_chamfer_loss:
            xyz = self._th_bt2xyz(th_bt)
            w_chamfer = 10  # 1
            chamfer_loss = chamfer_dist(xyz, y)
            output['log_p'] = output['log_p'] - w_chamfer * chamfer_loss

        if return_dict:
            return output
        else:
            raise NotImplementedError

    def log_prob(
            self, y: dict, x, div_type=0, mods=None, return_dict=True
            ) -> Union[dict, torch.Tensor]:
        return self._reverse_kld(y, x, mods=mods, return_dict=return_dict)

    def get_loss(self, x, y: dict, div_type=0, mods=None, return_dict=True):
        if self._extra_ws['log_p_vis_giv_z'] == 0.:
            # PT p_vis_giv_z.
            return self._train_p_vis_giv_z(y, x)

        return self.log_prob(
            y, x, div_type=div_type, mods=mods, return_dict=return_dict)

    def sample(self, x, N: Union[int, list] = 5, temp=0.5, mods=None, y=None) -> dict:
        """ #TODO: sample + GA --> multi hypos --> best

        Returns:
            output:
                - th_bt(s): shape: (N, B, Th + Bt)
                - logs_t (Optional)
                - xyz, uv, mesh
                % - sigma_i
        """
        if type(N) == list:
            N, N_quant = N
        output = {}
        output['image'] = y['image']  # for viz

        if isinstance(self.feat_extractor, BasicEnc):
            _, feat, _1 = self.feat_extractor(x)
        bs = x.shape[0]
        z = self._sample_q_z_giv_i(feat, N=N, temp=temp, y=y, **{'sample': True})
        z = z.reshape(N, -1, *z.shape[1:])
        if N_quant < N:
            log_q = self._reverse_log_q(z.flatten(0, 1), feat.repeat(N, 1)).reshape(N, -1)
            indices = torch.topk(log_q, N_quant, dim=0)[1]  # shape: (Q, B)
            indices = indices[..., None].repeat(1, 1, z.shape[-1])
            z = torch.gather(z, 0, indices)
            N = N_quant
        output['th_bt'] = z[..., : 58]
        output['logs_t'] = z[..., -3:]
        if mods is None:
            mods = ['xyz', 'uv', 'verts']
        dec_out = self._th_bt_product(
            output['th_bt'], output['logs_t'], mods=mods, inv_norm=True)
        for mod in ['verts', 'xyz', 'uv']:
            if mod in mods:
                output[mod] = dec_out[mod].reshape(N, bs, -1)
        if 'verts' in mods:
            output['faces'] = self.mano_dec.mano_faces
        return output

    def training_step_start(self, step):
        kld_w_init, kld_w_steps = self.kld_w_annealing
        self.kld_w = kld_w_init + (self.kld_w_final - kld_w_init) * min(1., step / kld_w_steps)
