from re import M
import torch.nn as nn
import torch
import torch.nn.functional as F
from manopth.manolayer import ManoLayer as manoLayer
from utils import FreiHand2RHD_skeidx, RHD2FreiHand_skeidx, RHD2Bighand_skeidx
# import neural_renderer as nr
import numpy as np

class ManoLayer(nn.Module):
    def __init__(
            self, MANO_dir= './mano/', flat_hand_mean=True, ncomps=45,
            use_pca=False, n_latent=None, skeidx = 'FreiHand', output_size=256,
            mask_sz=256):
        super(ManoLayer, self).__init__()
        # center_idx: the choice does not matter.
        # MANO_dir: Freihand. Needs to remap for other datasets!
        # Notice, use flat_hand_mean=True to fit the MANO_SMPL interface!
        self.mano_layer = manoLayer(
            center_idx=9, flat_hand_mean=flat_hand_mean, ncomps=ncomps, side='right',
            mano_root=MANO_dir, use_pca=use_pca)
        self.Jreg = self.mano_layer.th_J_regressor
        self.n_latent = n_latent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.n_latent is not None:
            self.mano_beta = nn.Sequential(nn.Linear(self.n_latent,512),
                    nn.ReLU(),
                    nn.Linear(512,10)
            )
            self.mano_theta = nn.Sequential(nn.Linear(self.n_latent,512),
                            nn.ReLU(),
                            nn.Linear(512,48)
            ) 
        self.skeidx  = skeidx # default skeleton idx order

        self.output_size = output_size
        self.mask_sz = mask_sz
        self.mano_faces = self.mano_layer.th_faces
        self.f = (self.mano_faces[None, :, :]).int()
        # self.renderer = nr.Renderer(perspective=False, image_size=self.mask_sz, camera_mode='look_at',anti_aliasing=True)
        # np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        # np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
        # self.base_rot_mat_x = torch.from_numpy(np_rot_x).float()

    def forward(self, z=None, beta=None, theta=None):
        if beta is None:
           beta = self.mano_beta(z) # bs*10
        if theta is None:
           theta = self.mano_theta(z) #bs*48
        beta = beta.view(-1,10)
        theta = theta.view(-1,48)
        mano_verts, mano_joints = self.mano_layer(theta, beta) # bs*778*3
        joints = self.xyz_from_vertice(mano_verts).permute(1,0,2) # bs*21*3  # it should be FreiHand idx
        if self.skeidx == 'RHD':
            joints = joints[:, FreiHand2RHD_skeidx, :]
            mano_joints = mano_joints[:, FreiHand2RHD_skeidx, :]
        elif self.skeidx == 'BigHand':
            joints = joints[:, FreiHand2RHD_skeidx, :][:, RHD2Bighand_skeidx, :]
            mano_joints = mano_joints[:, FreiHand2RHD_skeidx, :][:, RHD2Bighand_skeidx, :]
        return {'beta':beta, 'theta':theta, 'mesh': mano_verts, 'joints': joints, 'mano_joints': mano_joints}

    def render(
            self, scale_camera, trans_camera, vertex=None, norm=None,
            render=['mask']):
        """
        Args:
            vertex: normed rel
        """
        # we should limit the range of scale and trans
        # -1<trans<1 and scale>0
        scale_camera = scale_camera.abs()

        out = dict()
        # 3D camera coordinates to 2D image coordinates using orthographic projection (scale and trans) 
        # uv_pred = self.batch_orth_proj(joint, scale_camera, trans_camera, self.output_size)
        # out['uv'] = uv_pred.float()

        if vertex is not None:
            v = vertex.clone()
            v_shape = v.shape

            # v[:, :, :2] =  (scale_camera * self.output_size * v[:, :, :2].reshape(v_shape[0], -1)).reshape(v_shape[0],v_shape[1],2)
            # v[:, :, :2] = v[:, :, :2] + trans_camera.reshape(-1, 1, 2) * self.output_size / 2 + self.output_size / 2 
            # v[:, :, :2] = 2 * v[:, :, :2] / self.output_size - 1  # normalize the value to [-1,1]
            v[:, :, : 2] = self.batch_orth_proj(vertex, scale_camera, trans_camera, self.output_size, inv_norm=False)
            
            # For vertex render, note that the coordinate system of MANO is not consistent with that of neural_renderer
            v[:, :, 1] *= - 1  # coordinate inconsistent as mentioned above

            v[:, :, 2] = vertex[:, :, 2] * norm[:, None]
            v[:, :, 2] = v[:, :, 2]/1000.0  # convert mm to m, otherwise we will fail to project

            # z_value = v[:, :, 2].clone().view(-1,v_shape[1])
            # z_offset = torch.min(z_value,-1)[0]
            # v[:, :, 2] -= z_offset.view(-1,1)
            
            # depth or mask render based on neural_renderer
            v_shape = v.shape
            face = self.f.repeat(v_shape[0],1,1).to(self.device)
            if 'mask' in render:
                out['mask'] = self.renderer.render_silhouettes(v, face)
            if 'depth' in render:
                out['depth'] = self.renderer.render_depth(v, face)

        return out


    # change the order of 16 keypoints and get 5 tips from mesh
    @staticmethod
    def get_keypoints_from_mesh_np(mesh_vertices, keypoints_regressed):
        """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
        kpId2vertices = {
            4: [744],  # ThumbT
            8: [320],  # IndexT
            12: [443],  # MiddleT
            16: [555],  # RingT
            20: [672]  # PinkT
        }
        keypoints = [0.0 for _ in range(21)]  # init empty list

        # fill keypoints which are regressed
        mapping = {0: 0,  # Wrist
                1: 5, 2: 6, 3: 7,  # Index
                4: 9, 5: 10, 6: 11,  # Middle
                7: 17, 8: 18, 9: 19,  # Pinky
                10: 13, 11: 14, 12: 15,  # Ring
                13: 1, 14: 2, 15: 3}  # Thumb

        for manoId, myId in mapping.items():
            keypoints[myId] = keypoints_regressed[:,manoId, :]

        # get other keypoints from mesh
        for myId, meshId in kpId2vertices.items():
            keypoints[myId] = torch.mean(mesh_vertices[:,meshId, :], 1)

        #print(np.array(keypoints).shape,keypoints[0].shape )
        keypoints = torch.stack(keypoints)
        return keypoints

    # get keypoints from mesh, this is only true for freihand
    def xyz_from_vertice(self, vertice):
        J_regressor = self.Jreg.t().float().to(vertice.device)
        joint_x = torch.matmul(vertice[:, :, 0], J_regressor)
        joint_y = torch.matmul(vertice[:, :, 1], J_regressor)
        joint_z = torch.matmul(vertice[:, :, 2], J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
        coords_kp_xyz3 = self.get_keypoints_from_mesh_np(vertice, joints)
        return coords_kp_xyz3

    @staticmethod
    def batch_orth_proj(
            joint, scale_camera, trans_camera, image_size: int = 256, inv_norm=True):
        """
        Args:
            joint: shape: (NB, K, 3)
            scale_camera: shape: (NB, 1)
            trans_camera: shape: (NB, 2)
        
        Returns:
            out: shape: (NB, K, 2)
        """
        out = scale_camera[:, None, :] * joint[:, :, : 2] + trans_camera[:, None, :]
        if inv_norm:  # --> image space
            out = (out + 1.) / 2. * image_size
        return out


if __name__ == '__main__':
    import os
    mano = ManoLayer()
    bs = 5
    rand = torch.rand(bs,512)
    out = mano(rand)
    # print(out['mesh'])
    print(out['joints'].shape)
    print(out['mano_joints'].shape)

    scale = torch.ones(1).to('cuda')*0.004
    trans = torch.zeros(bs,2).to('cuda')
    render_out = mano.render(scale, trans, out['joints'].to('cuda'), out['mesh'].to('cuda'),)
    uv, mask = render_out['uv'], render_out['mask']
    print(uv.shape)
    print(mask.shape)


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask = mask[0].squeeze().detach().cpu().numpy()
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    pose2d = uv[0].squeeze().detach().cpu().numpy()


    plt.figure()
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(mask)
    plt.scatter(pose2d[:, 0], pose2d[:, 1], alpha=0.6)
    plt.savefig('render_test.jpg')
    plt.close()

    import trimesh
    test_mesh = trimesh.Trimesh(vertices=out['mesh'][0].squeeze().cpu().detach().numpy(),
                            faces=mano.mano_faces.squeeze().cpu().detach().numpy())
    test_mesh.export('mesh.obj')

