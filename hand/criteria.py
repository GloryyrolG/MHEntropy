import numpy as np
import torch
from torch import nn
import trimesh
from utils import align_w_scale, meanEuclideanLoss

from matplotlib import pyplot as plt
import trimesh
from viz import (
    plot_pose3d, colorlist_gt, viz_2djoints, export_pose3d_gif, mesh_axis_tsfm)


def kl_loss(z_mean, z_stddev, goalStd=1.0):
    latent_loss = 0.5 * torch.sum(z_mean ** 2 + z_stddev ** 2 - torch.log(z_stddev ** 2) - goalStd, 1)
    return latent_loss


def chamfer_dist(norm_rel_xyz, target: dict):
    """
    Args:
        norm_rel_xyz: shape: ((N,) B, K, 3)
    """
    # Chamfer.
    abs_xyz = norm_rel_xyz * target['scale'][:, None, None] * 1000
    abs_xyz = abs_xyz + target['original_pose3d'][:, [12]]
    if len(norm_rel_xyz.shape) == 3:
        abs_xyz = abs_xyz.unsqueeze(dim=0)
    B = target['scale'].shape[0]
    dists = (
        (abs_xyz[:, :, :, None, :]
         - target['object_verts'].reshape(B, -1, 3)[:, None, :, :]  # shape: (N, B, K, VO, 3)
        ).norm(p=2, dim=-1))  # * target['object_vertns'].reshape(B, -1, 3)[:, None, :, :]
    dists_o2h = dists.min(-1)[0].mean(-1)  # shape: (N, B)
    dists_h2o = dists.min(-2)[0].mean(-1)
    dist = dists_o2h + dists_h2o
    if len(norm_rel_xyz.shape) == 3:
        return dist[0]
    else:
        return dist


class MHEntLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super(MHEntLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, output, target):
        """ Returns what the output provides. Although it is called HMR, it
            supports eval of several settings including HPE
        
        Args:
            output: log_p, (xyz), (verts), (uv)
        """
        losses, metrics = {}, {}
        losses['neg_log_p'] = -output['log_p']

        if 'xyz' in output:
            N, B = output['xyz'].shape[: 2]
        else:
            N, B = output['uv'].shape[: 2]

        aligned = False
        unaligned_output = {}
        lbl_ses = []
        for lbl_s in ['xyz', 'verts']:
            if lbl_s in output:
                lbl_ses.append(lbl_s)
                unaligned_output[lbl_s] = output[lbl_s]
        if aligned:
            aligned_output = {}
            for lbl_s in lbl_ses:
                aligned_output[lbl_s] = output[lbl_s].clone().cpu().numpy()
            for lbl_s in lbl_ses:
                _target = target['pose3d'] if lbl_s == 'xyz' else target.get('verts', None)
                if _target is None:
                    continue
                for n in range(N):
                    for b in range(B):
                        mtx2_t, R, s, s1, s2, t1, t2 = align_w_scale(
                            _target[b].reshape(-1, 3).cpu().numpy(),
                            output[lbl_s][n, b].reshape(-1, 3).cpu().numpy(),
                            return_trafo=True)
                        aligned_output[lbl_s][n, b] = mtx2_t.reshape(-1)
            for lbl_s in lbl_ses:
                output[lbl_s] = torch.from_numpy(aligned_output[lbl_s]).float().cuda()

        chamfer_select = False
        if chamfer_select:
            dist = chamfer_dist(output['xyz'].reshape(N, B, -1, 3), target)
        
        if 'xyz' in output:
            xyz = torch.flatten(output['xyz'], 0, 1)
            eucLoss_3d_rgb = meanEuclideanLoss(
                xyz, target['pose3d'].repeat(N, 1), target['scale'].repeat(N),
                reduction='none').reshape(N, B, -1)  # shape: (N, B, K)
        uv_gt = (target['crop_uv'] + 1.) / 2. * 256
        if 'uv' in output:
            uv_pred = output['uv']
        else:
            # Uses GT s and t.
            _xyz = output['xyz'].reshape(*output['xyz'].shape[: -1], -1, 3)
            uv_pred = target['st'][:, None, [0]] * _xyz[..., : 2] + target['st'][:, None, -2:]
            uv_pred = (uv_pred + 1) / 2 * 256
            output['uv'] = uv_pred = uv_pred.flatten(start_dim=-2)
        eucLoss_2d_rgb = (uv_pred - uv_gt).reshape(N, B, -1, 2).norm(p=2, dim=-1)

        weights = {
            'sample': torch.ones_like(target['vis']),  # all kps
            'vis': (target['vis'] == 1.).float(),  # vis. (HMDN)
            'invis': (target['vis'] != 1.).float(),  # occ. (HMDN)
        }
        root_idx = 12  # int(target['_root_idx'][0])
        weights['vis'][:, root_idx] = 0.  # not count the root
        weights['invis'][:, root_idx] = 0.

        def _group_stats(stats, weight):
            """
            Args:
                stats: shape: ((N,) B, K)
                weight: shape: ((N,) B, K)

            Returns:
                mpj: shape: ((N,) B)
            """
            num_vis = weight.sum(-1)
            mpj = (stats * weight).sum(-1) / (num_vis + 1e-16)  # shape: ((N,) B)
            # mpj[num_vis == 0.] = 0.  # for B
            if len(num_vis.shape) == 2:
                num_vis = num_vis[0]
            num_valid = (num_vis > 0.).sum().item()  # shape: (B,)
            mpj = mpj * B / (num_valid + 1e-16) if num_valid else mpj * 0.
            return mpj

        # Mean Per Joint EPE.
        sup_ses = ['3d', '2d']

        for sup_s in sup_ses:
            eucLoss_rgb = eval(f'eucLoss_{sup_s}_rgb')
            D = int(sup_s[0])
            if sup_s == '3d':
                coord = unaligned_output['xyz'] * target['scale'][:, None]  # unnormed
            else:  # 2d
                coord = output['uv']
            for attr, weight in weights.items():
                key = f'eucLoss_{sup_s}_rgb_{attr}'
                # Notice the calc order!
                mpjpe = _group_stats(eucLoss_rgb, weight[None, ...].repeat(N, 1, 1))
                if sup_s == '2d' and attr == 'vis':
                    # metrics[key] = mpjpe.min(0)[0]  # BH
                    metrics[key] = mpjpe.max(0)[0]  # WH
                else:
                    metrics[key] = mpjpe.min(0)[0]  # BH

			    # Along the sample dim.
                coord = coord.reshape(N, B, -1, D)
                if N == 1:
                    spspe = torch.zeros(B, coord.shape[-2], dtype=torch.float32,
                                        device='cuda')
                else:
                    spspe = coord.std(0).prod(-1)  # shape: (B, K). Measure of an ellipsoid
                spspe = spspe ** (1/D) * (D ** 0.5)  # dim unity: V --> l
                metrics[f'{key}_std'] = _group_stats(spspe, weight)
                # Mean.
                mpspe = None
                if attr == 'vis':
                    mpspe = eucLoss_rgb.mean(0)  # (NF)
                if mpspe is not None:
                    metrics[f'{key}_mean'] = _group_stats(mpspe, weight)
        if False:  # 'verts' in output and 'verts' in target:
            metrics['eucLoss_mesh_rgb_sample'] = (  # mesh BH
                output['verts'] - target['verts']).reshape(N, B, -1, 3).norm(p=2, dim=-1).mean(-1).min(0)[0]

        return sum([v.mean() for v in losses.values()]), losses, metrics
