from __future__ import print_function, unicode_literals
import sys
from typing import Iterable, Optional, Tuple

from utils import align_w_scale
sys.path.append('..')
import pickle,os,imageio,torchvision,copy
from dataloader.dataPreprocess.preprocess import *
import dataloader.dataPreprocess.augment as augment
from torch.utils.data import Dataset
from torchvision.transforms.functional import erase
from matplotlib import pyplot as plt

# from openpose.hand import Hand


# SET THIS to where RHD is located on your machine
path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = './RHD_published_v2'
if(not os.path.exists(path_to_db)):
    path_to_db = './dataset/RHD_published_v2'


def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

class RHDDateset3D(Dataset):
    def __init__(self, mode='training', path_name=path_to_db, view_correction=True, uv_norm=True):
        self.mode = mode
        self.image_list = []
        self.mask_list = []
        self.depth_list = []
        self.image_uv = []
        self.image_xyz = []
        self.kp_visible_list = [] # visibility of the keypoints, boolean
        self.camera_intrinsic_matrix = []  # matrix containing intrinsic parameters
        self.row = 320
        self.col = 320
        self.num_samples=0
        self.path_name=path_name
        self.mode=mode
        self.view_correction = view_correction
        self.uv_norm = uv_norm
        self.data_aug = True
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())
        print('self.num_samples',self.num_samples)

        # OpenPose.
        if False:
            self.hand_estimation = Hand('../pytorch-openpose/model/hand_pose_model.pth')
            self.openpose = [None for _ in range(self.num_samples)]
            openpose_pth = f'./tmp/openpose_{int(mode == "evaluation")}.pkl'
            if os.path.isfile(openpose_pth):
                print(f"> Loadin {openpose_pth}")
                with open(openpose_pth, 'rb') as f:
                    self.openpose = pickle.load(f)
            else:
                for idx in range(30):  # generate in advance
                    self.__getitem__(idx)
                with open(openpose_pth, 'wb') as f:
                    pickle.dump(self.openpose, f)

    def __len__(self):
        return self.num_samples
        # return 5000

    def __getitem__(self, idx):
        # idx = idx % 10  # for debug

        ori_idx = idx
        if(idx==20500 or idx==28140):idx=0
        if self.mode == 'evaluation' and idx in [1012, 1324]:  # bad images
            idx = 0
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            anno=self.anno_all[idx]
            image = imageio.imread(os.path.join(path_to_db, self.mode, 'color', '%.5d.png' % idx))
            mask = imageio.imread(os.path.join(path_to_db, self.mode, 'mask', '%.5d.png' % idx))
            depth = imageio.imread(os.path.join(path_to_db, self.mode, 'depth', '%.5d.png' % idx))
            depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])


            # get info from annotation dictionary
            kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel. 320
            kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
            kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

            vis = vis_0 = check_occlusion(
                kp_coord_uv, depth, kp_coord_xyz, delta=0.02, quant=2)

            root_idx = 12
            image_crop, depth_crop, cloud_normed, pose3d_normed, cloud_vc_normed, \
            pose3d_vc_normed, viewRotation, scale, hand_side, heatmap,\
            (crop_center, crop_size, pose3d_root), (s1, t1), crop_uv, uv_vis,\
            hand_mask\
                = preprocessSample(image, depth, mask, kp_coord_uv, kp_visible, kp_coord_xyz, camera_intrinsic_matrix)
            # Can have a deviation of annotations cuz of quant (e.g.,
            # depth_crop[round(crop_uv[:, 1]), round(crop_uv[:, 0])]).

            vis = vis[: 21] if hand_side[0] else vis[21:]

            image_crop = image_crop.reshape([256, 256, 3])
            depth_crop = depth_crop.reshape([256, 256, 1])
            cloud_normed = cloud_normed.reshape([4000, 3])
            cloud_vc_normed = cloud_vc_normed.reshape([4000, 3])
            heatmap = heatmap.reshape([64, 64, 21])

            # OpenPose. Bef data aug.
            if False:
                # Cannot afford to run the annotation every time. If only
                # considering the image appearance, it is sufficient to run it
                # once before data aug.
                if self.openpose[ori_idx] is None:  # can only enter once
                    pseudo_uv = openpose_annot(
                        image_crop[:, :, [2, 1, 0]],  # RGB --> BGR
                        hand_estimation=self.hand_estimation)[0]  # already flipped to right
                    self.openpose[ori_idx] = pseudo_uv # store openpose
                else:
                    pseudo_uv = self.openpose[ori_idx]
                vis = pseudo_uv[:, 0] != 0  # override vis
                crop_uv = pseudo_uv  # override uv to perf data aug

            # Patch occlusion. Bef data aug.
            if True:
            # if idx < self.num_samples // 2:  # mix training
                image_crop, vis, (patch_cx, patch_cy, patch_r, object_mask) = patch_occlusion(
                    image_crop, size=50, vis=None, idx=idx, crop_uv=crop_uv)
            else:
                vis = np.ones((21,), dtype=np.float32)
                patch_cx, patch_cy, patch_r  = 0, 0, 0
                object_mask = np.zeros(image_crop.shape[: -1])

            if self.view_correction:
                cloud_output = copy.deepcopy(cloud_vc_normed)
                pose3d_output = copy.deepcopy(pose3d_vc_normed)
            else:
                cloud_output = copy.deepcopy(cloud_normed)
                pose3d_output = copy.deepcopy(pose3d_normed)

            rotMat = np.eye(2, 3)
            if(self.mode=='training' and self.data_aug):
                image_crop, depth_crop, cloud_output, _, pose3d_output, crop_uv_output, rotMat, hand_mask, object_mask = \
                    augment.processing_augmentation(image_crop, depth_crop, cloud_output, heatmap, pose3d_output, hand_side, crop_uv, hand_mask, object_mask)
                cloud_output = cloud_output.transpose(1, 0)
                # hue=0.8
                cjitter=torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4,1.6], saturation=[0.4,1.6], hue=0.1)
                image_trans = torchvision.transforms.Compose([cjitter,torchvision.transforms.ToTensor()])
            else:
                image_crop, depth_crop, cloud_output, _, pose3d_output, crop_uv_output, hand_mask, object_mask = \
                    augment.processing(image_crop, depth_crop, cloud_output, heatmap, pose3d_output, hand_side, crop_uv, hand_mask, object_mask)
                cloud_output = cloud_output.transpose(1, 0)
                image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

            # vis = check_occlusion(
            #     crop_uv_output.reshape(-1, 2),
            #     (1. - depth_crop[:, :, 0]) * 5.,  # depth has been normed
            #     pose3d_output.reshape(-1, 3) * scale + pose3d_root,
            #     delta=0.03)
            # Within-the-boundary.
            wib = check_wib(crop_uv_output.reshape(-1, 2), depth_crop.shape[: 2], quant=2)
            # vis = np.logical_and(vis, wib)
            # 0.: patch;
            # 1.: vis;
            # 2.: wib.
            vis[np.logical_and(vis == 1., wib == 0.)] = 2.

            # Patch.
            patch_center = np.array([patch_cx, patch_cy], dtype=np.float32)
            if self.mode == 'training' and self.data_aug:
                patch_center = rotMat[:, : 2] @ patch_center + rotMat[:, 2]
            if hand_side[0]:
                patch_center[0] = 255 - patch_center[0]
            patch_size = np.array([patch_r], dtype=np.float32)
            patch = np.concatenate([patch_center, patch_size])

            if self.uv_norm:
                crop_uv_output = crop_uv_output / 256.0 * 2. - 1.
                patch[: 2] = patch[: 2] / 256 * 2. - 1.
                patch[2] = patch[2] / 256 * 2.

            # get rot_mat_inv
            rot_mat_inv = np.eye(3)
            rot_mat_inv[:2, :] = rotMat
            rot_mat_inv = np.linalg.inv(rot_mat_inv.T)
            rot_mat_inv = rot_mat_inv[:, :2]

            st = compute_st(pose3d_output, crop_uv_output)
            hand_mask = cv2.resize(hand_mask, (64, 64), interpolation=cv2.INTER_NEAREST)

            target = {}
            target['cloud'] = cloud_output
            target['pose3d'] = pose3d_output.astype(np.float32)
            target['scale'] = np.float32(scale)
            target['viewRotation'] = viewRotation
            # kwargs for xyz2crop.
            target.update({
                'crop_uv': crop_uv_output.astype(np.float32),
                'target_uv_weight': uv_vis,
                'crop_center': crop_center,
                'crop_size': np.float32(crop_size),
                'hand_side': hand_side[0].astype(np.float32),
                'bone_length': target['scale'],
                'pose3d_root': pose3d_root,
                'camera': camera_intrinsic_matrix,
                'rot_mat_inv': rot_mat_inv.astype(np.float32),
                'original_pose3d': kp_coord_xyz[: 21] if hand_side[0] else kp_coord_xyz[-21:],
                '_rot_mat': (rotMat[:, : 2] / np.linalg.norm(rotMat[0, : 2])).astype(np.float32),
                'uvd': np.concatenate([crop_uv_output.reshape(21, 2), pose3d_output.reshape(21, 3)[:, [-1]]], axis=1).ravel(),
                
                'dataset': 'rhd',

                'st': st.astype(np.float32),
                '_idx': ori_idx,
                '_split': int(self.mode == 'evaluation'),
                'mask': hand_mask,
                'vis': vis.astype(np.float32),
                'patch': patch,
                'object_mask': object_mask,
            })
            for k, v in target.items():
                if isinstance(v, np.ndarray):
                    target[k] = torch.from_numpy(v).float()
                elif isinstance(v, np.float32):
                    target[k] = torch.tensor(v)

            return image_crop, target


def compute_st(pose3d, crop_uv, root_idx=12):
    """ Orth proj, uv=s{normed_rel_xyz}[:2]+t

    Args:
        pose3d (np.ndarray): shape: (3K,). Normed rel
        crop_uv (np.ndarray): shape: (2K,).
    
    Returns:
        st (np.ndarray)
    """
    if len(pose3d.shape) == 2:
        pose3d = pose3d.reshape(-1)
    if len(crop_uv.shape) == 2:
        crop_uv = crop_uv.reshape(-1)
    # Compute s and t in uv. Might be prob?!
    # s_uv = np.linalg.norm(
    #     crop_uv.reshape(-1, 2)[norm_idx]
    #     - crop_uv.reshape(-1, 2)[root_idx])
    # s_uvd = np.linalg.norm(
    #     pose3d.reshape(-1, 3)[norm_idx, : 2]
    #     - pose3d.reshape(-1, 3)[root_idx, : 2])
    # s = s_uv / s_uvd

    _, R, s, s1, s2, t1, t2 = align_w_scale(
        crop_uv.reshape(-1, 2), pose3d.reshape(-1, 3)[:, : 2], return_trafo=True)
    # print(f"R = {R}")
    t = -t2 / s2 * s * s1 + t1
    s *= s1 / s2
    # t_ = (crop_uv.reshape(-1, 2)[root_idx]
    #      - s * pose3d.reshape(-1, 3)[root_idx, : 2])
    # print(t_ - t)
    st = np.concatenate([np.array([s]), t])
    return st


def check_wib(uv_coordinate: np.ndarray, range_: Iterable, quant=1):
    num_joints = uv_coordinate.shape[0]
    visibles = np.zeros(num_joints)
    for i in range(num_joints):
        x_i, y_i = round(uv_coordinate[i][1]), round(uv_coordinate[i][0])
        for x in range(x_i - quant + 1, x_i + quant):
            for y in range(y_i - quant + 1, y_i + quant):
                if x < 0 or y < 0 or x > range_[1]-1 or y > range_[0]-1:
                    continue
                visibles[i] = 1
                break
            if visibles[i]:
                break
    return visibles


def check_occlusion(uv_coordinate, depthmap, pose3d, delta=0.1, quant=1):
    '''
    uv_coordinate: (num_joints, 2)
    depthmap: the size should match uv_coordinate
    pose3d: (x,y,z) of the joint, camera coordinate, but z is good
    '''
    num_joints = uv_coordinate.shape[0]
    visibles = np.zeros(num_joints)
    for i in range(num_joints):
        x_i, y_i = round(uv_coordinate[i][1]), round(uv_coordinate[i][0])
        for x in range(x_i - quant + 1, x_i + quant):
            for y in range(y_i - quant + 1, y_i + quant):
                if x < 0 or y < 0 or x > depthmap.shape[1]-1 or y > depthmap.shape[0]-1:
                    continue
                depth_i = depthmap[x][y]
                joint_depth_i = pose3d[i][2]
                if abs(depth_i - joint_depth_i) > delta:
                    continue
                visibles[i] = 1
                break
            if visibles[i]:
                break
    return visibles


def openpose_annot(oriImg: np.ndarray, is_left: bool = False, hand_estimation=None) -> np.ndarray:
    all_hand_peaks = []
    hands_list = [[0, 0, oriImg.shape[0], is_left]]  # currently only supports single hand
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        if is_left:  # rev order
            peaks_t = peaks.copy()
            peaks_t[5: 9] = peaks[17: 21]
            peaks_t[9: 13] = peaks[13: 17]
            peaks_t[13: 17] = peaks[9: 13]
            peaks_t[17: 21] = peaks[5: 9]
            peaks = peaks_t
        all_hand_peaks.append(peaks)
    return all_hand_peaks


def patch_occlusion(
        image_crop: np.ndarray, size=100, vis: Optional[np.ndarray] = None,
        idx=0, crop_uv: Optional[np.ndarray] = None, dataset='RHD'
        ) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Args:
        idx: for pesudo rand gen
        crop_uv: shape: (K, 2)
    """
    K = crop_uv.shape[0]
    rnd_patchtype = 1  # 2
    ori_image_crop = image_crop
    image_crop = image_crop.copy()
    if rnd_patchtype in [0, 1]:
        if rnd_patchtype == 1:
            # Type 2. Centering at a KP.
            rnd_kpidx = (
                [2, 6, 10, 14, 18][idx % 5])
                # idx % K)
            cx, cy = crop_uv[rnd_kpidx].astype('int')
        else:
            cx, cy = image_crop.shape[1] // 2, image_crop.shape[0] // 2
        # No stoc.
        # cx = cx + torch.randint(-40, 40, size=(1,)).item()
        # cy = cy + torch.randint(-40, 40, size=(1,)).item()

        # Rect.
        # h, w = size, size
        # image_crop[max(0, cy - w // 2): cy + w // 2, max(0, cx - h // 2): cx + h // 2] = 0
        # uv_t = crop_uv
        # occ_mask = np.logical_and(np.logical_and(np.logical_and(
        #     uv_t[:, 0] >= cx - h // 2,
        #     uv_t[:, 0] <= cx + h // 2),
        #     uv_t[:, 1] >= cy - w // 2),
        #     uv_t[:, 1] <= cy + w // 2)
        # Circ.
        r = size
        xx, yy = np.meshgrid(np.arange(image_crop.shape[1]), np.arange(image_crop.shape[0]))
        circ_fn = lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 - r ** 2
        occ_maskm = circ_fn(xx, yy) <= 0
        image_crop[occ_maskm] = 0
        occ_mask = circ_fn(crop_uv[:, 0], crop_uv[:, 1]) <= 0

        # image_crop = ori_image_crop
        
    elif rnd_patchtype == 2: 
        # Has probs.
        rnd_fidx = 3  # idx % 5
        rnd_r = idx % 2
        if dataset == 'RHD':
            start_fkpidx = 1 + rnd_fidx * 4
            end_fkpidx = 1 + (rnd_fidx + 1) * 4
        else:
            raise NotImplementedError
        msked_kpuvs = crop_uv[start_fkpidx: end_fkpidx]
        cart = msked_kpuvs - crop_uv[0]
        phi = np.arctan2(cart[:, 1], cart[:, 0])
        rot_argmax = phi.argmax()
        rot_argmin = phi.argmin()
        k = msked_kpuvs[rot_argmax] - msked_kpuvs[rot_argmin]
        k = k[1] / (k[0] + 1e-16)
        line = lambda x, y: (y - msked_kpuvs[rot_argmin, 1]) - k * (x - msked_kpuvs[rot_argmin, 0])
        line_kp0 = line(crop_uv[0, 0], crop_uv[0, 1])
        h = 20
        h = h * (k ** 2 + 1.) ** 0.5
        xx, yy = np.meshgrid(np.arange(image_crop.shape[1]), np.arange(image_crop.shape[0]))
        image_crop[(line(xx, yy) * np.sign(line_kp0 + 1e-16)) < h] = 0
        occ_mask = (line(crop_uv[:, 0], crop_uv[:, 1]) * np.sign(line_kp0 + 1e-16)) < h
        
    vis = vis.copy() if vis is not None else np.ones((crop_uv.shape[0],))
    # vis[:] = 1.  # override, only consider occlusion
    vis[occ_mask] = 0.

    return image_crop, vis, (cx, cy, r, occ_maskm.astype('float'))
