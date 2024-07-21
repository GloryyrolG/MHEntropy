import sys
import os
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch,random,logging
import numpy as np
from enum import Enum
from dataloader.dataPreprocess.preprocess import xyz2uvd
from scipy.linalg import orthogonal_procrustes

RHD2Bighand_skeidx = [0, 4, 8, 12, 16, 20, 3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, 19, 18, 17]
mano2Bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]
STB2Bighand_skeidx = [0, 17, 13, 9, 5, 1, 18, 19, 20, 14, 15, 16, 10, 11, 12, 6, 7, 8, 2, 3, 4]
FreiHand2RHD_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
Bighand2RHD_skeidx = [0, 8, 7, 6, 1, 11, 10, 9, 2, 14, 13, 12, 3, 17, 16, 15, 4, 20, 19, 18, 5]
RHD2FreiHand_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
Bighand2mano_skeidx = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]


def meanEuclideanLoss(pred, gt, scale, reduction='mean'):
    """ At the original scale"""
    pred = pred.view([pred.shape[0], -1, 3])
    gt = gt.reshape([pred.shape[0], -1, 3])
    eucDistance = torch.squeeze(torch.sqrt(torch.sum((pred - gt) ** 2, dim=2)))
    eucDistance = eucDistance * torch.squeeze(scale).view(scale.shape[0], 1)
    if reduction == 'mean':
        return torch.mean(eucDistance)
    else:
        return eucDistance

def pose_loss(p0, p1, scale, reduction='mean'):
    """
    Args:
        p0: shape: (B, KD)

    Returns:
        pose_loss_rgb: shape: [B]. SSE (Sum Squared Error).
        eucLoss_rgb: shape: []! It is a metric over a batch.
    """
    # pose_loss_rgb = torch.sum((p0 - p1) ** 2, 1)
    pose_loss_rgb = torch.sum((p0 - p1).abs(), 1)
    eucLoss_rgb = meanEuclideanLoss(p0, p1, scale, reduction=reduction)
    return pose_loss_rgb, eucLoss_rgb

def batch_normalize_pose3d(
        pose3d, root_idx, norm_idx=None, return_depth=False, return_st=False):
    """
    Args:
        pose3d: shape: (B, K, 3)
    """
    pose3d_root = pose3d[:, root_idx, :].clone().view(-1,1,3)  # this is the root coord
    pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
    pose3d_output = pose3d_rel.clone()
    if norm_idx is not None:
        bone_length = torch.sqrt(torch.sum(pose3d_rel[:, norm_idx, :] ** 2, -1)).view(-1,1,1)
        pose3d_normed = pose3d_rel / bone_length
        pose3d_output = pose3d_normed.clone()
        relative_depth = pose3d_normed[:, :, 2].view(-1,21,1)
    if not return_depth:
        if not return_st:
            return pose3d_output
        else:
            return pose3d_output, pose3d_root, bone_length[:, 0, 0]
    else:
        return pose3d_output, relative_depth

def init_fn(worker_id):np.random.seed(worker_id)

class Mode(Enum):
    Train = 1
    Eval = 2
    Refine = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        n = int(val != 0)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def setSeed(seed=None):
    import time
    if not isinstance(seed, int):
        seed = int(time.time())%10000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

import random,string
def rand_model_name() -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))


def xyz2crop(pose3d: torch.Tensor, target, resized_size=256, camera_type = 'perspective', root_idx=12, norm_idx=11):
    """
    checked.
    Currently, only for eval.

    Args:
        pose_3d: shape: (B, K, 3) or (B, 3K). It is scale-normalized.
        target (dict):.
        root_idx: Default: 12 for RHD.

    Returns:
        crop_uv (torch.Tensor):.
    """
    device = pose3d.device

    if len(pose3d.shape) == 2:
        pose3d = pose3d.view(pose3d.shape[0], 21, 3)

    if camera_type == 'perspective':
        pose3d = pose3d - pose3d[:, root_idx, :].unsqueeze(1).clone()
        crop_center, crop_size, hand_side, bone_length, pose3d_root, camera, rot_mat_inv = \
            target['crop_center'].to(device), target['crop_size'].to(device), target['hand_side'].to(device),\
            target['bone_length'].to(device),target['pose3d_root'].to(device), target['camera'].to(device), \
            target['rot_mat_inv'].to(device)

        assert len(bone_length.shape) == 1
        pose3d = pose3d * bone_length[:, None, None]  # scale up

        pose3d = pose3d + pose3d_root.unsqueeze(1).clone()
        pose3d = pose3d.view(-1, 21, 3)
        uvd = xyz2uvd(pose3d,camera)
        uv = uvd[:,:,:2]

        _, crop_d = batch_normalize_pose3d(pose3d, root_idx, norm_idx=norm_idx, return_depth=True)

        crop_scale = resized_size / (crop_size * 2)
        crop_uv = relocate_uv(uv, crop_center, resized_size, crop_scale)

        bs = pose3d.shape[0]
        for i in range(bs):
            # if hand_side[i]:
            if hand_side[i, 0] == 1.:  # change to fit current code
                crop_uv[i, :, 0] = resized_size - crop_uv[i, :, 0]
    else:
        raise NotImplementedError

    return crop_uv, crop_d


def relocate_uv(uv: torch.Tensor, crop_center, resized_size, crop_scale):
    keypoint_uv21_u = (uv[:, :, 0].view(-1,21,1) - crop_center[:,0].view(-1,1,1)) * crop_scale.view(-1,1,1) + resized_size // 2
    keypoint_uv21_v = (uv[:, :, 1].view(-1,21,1) - crop_center[:,1].view(-1,1,1)) * crop_scale.view(-1,1,1) + resized_size // 2
    coords_uv = torch.cat((keypoint_uv21_u, keypoint_uv21_v), dim=-1)
    return coords_uv


# From RLE
# PCK@
def calc_coord_accuracy(
        output, target, hm_shape=None, output_3d=False, num_joints=None,
        other_joints=None, root_idx=None, thr=0.5, ds_type='human',
        output_normalized=True):
    """ Calculate integral coordinates accuracy.

    Args:
        output: shape: (B, K, D)
    """
    if hm_shape is None:
        hm_shape = [64, 48, 0]

    coords = output.detach().cpu().numpy()
    coords = coords.astype(float)

    if output_3d:
        labels = target['pose3d']
        label_masks = torch.ones_like(target['pose3d'])
    else:
        labels = target['crop_uv']
        label_masks = target['target_uv_weight']

    if num_joints is not None:
        if other_joints is not None:
            coords = coords.reshape(coords.shape[0], num_joints + other_joints, -1)
            labels = labels.reshape(labels.shape[0], num_joints + other_joints, -1)
            label_masks = label_masks.reshape(label_masks.shape[0], num_joints + other_joints, -1)
            coords = coords[:, :num_joints, :3].reshape(coords.shape[0], -1)
            labels = labels[:, :num_joints, :3].reshape(coords.shape[0], -1)
            label_masks = label_masks[:, :num_joints, :3].reshape(coords.shape[0], -1)
        else:
            coords = coords.reshape(coords.shape[0], num_joints, -1)
            labels = labels.reshape(labels.shape[0], num_joints, -1)
            label_masks = label_masks.reshape(label_masks.shape[0], num_joints, -1)
            coords = coords[:, :, :3].reshape(coords.shape[0], -1)
            labels = labels[:, :, :3].reshape(coords.shape[0], -1)
            label_masks = label_masks[:, :, :3].reshape(coords.shape[0], -1)

    if output_3d:
        hm_width, hm_height, hm_depth = hm_shape
        coords = coords.reshape((coords.shape[0], -1, 3))
    else:
        hm_width, hm_height = hm_shape[:2]
        coords = coords.reshape((coords.shape[0], -1, 2))
    
    # Normalize.
    if output_normalized:
        coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
        coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    if output_3d:
        labels = labels.cpu().data.numpy().reshape(coords.shape[0], -1, 3)
        label_masks = label_masks.cpu().data.numpy().reshape(coords.shape[0], -1, 3)

        if output_normalized:
            labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
            labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height
            labels[:, :, 2] = (labels[:, :, 2] + 0.5) * hm_depth

            coords[:, :, 2] = (coords[:, :, 2] + 0.5) * hm_depth

        if root_idx is not None:
            labels = labels - labels[:, root_idx, :][:, None, :]
            coords = coords - coords[:, root_idx, :][:, None, :]

    else:
        labels = labels.cpu().data.numpy().reshape(coords.shape[0], -1, 2)
        label_masks = label_masks.cpu().data.numpy().reshape(coords.shape[0], -1, 2)

        labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
        labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height

    num_joints = coords.shape[1]

    coords = coords * label_masks
    labels = labels * label_masks

    if output_3d:
        norm = np.ones((coords.shape[0], 3))
        if ds_type == 'human':
            norm = norm * np.array([hm_width, hm_height, hm_depth]) / 10
    else:
        norm = np.ones((coords.shape[0], 2))
        if ds_type == 'human':
            norm = norm * np.array([hm_width, hm_height]) / 10

    dists = calc_dist(coords, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i], thr=thr)
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_dist(preds, target, normalize):
    """ Calculate normalized distances

    Args:
        preds: shape: (B, K, D)
    """
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, thr=15. / 40.):  # 0.5
    """ Calculate accuracy with given input distance.

    Args:
        dists: shape: (B, K)
    """
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


# From RLE.
def evaluate_mAP(res_file, ann_type='bbox', ann_file='person_keypoints_val2017.json', silence=True):
    """Evaluate mAP result for coco dataset.

    Parameters
    ----------
    res_file: str
        Path to result json file.
    ann_type: str
        annotation type, including: `bbox`, `segm`, `keypoints`.
    ann_file: str
        Path to groundtruth file.
    silence: bool
        True: disable running log.

    """
    class NullWriter(object):
        def write(self, arg):
            pass

    ann_file = os.path.join('/data/coco/annotations/', ann_file)

    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)

    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if silence:
        sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]

    return info_str


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def heatmap_to_coord(
        pred_jts, pred_scores, hm_shape, bbox: list, output_3d=False):
    """
    Args:
        bbox: shape: (4,)
    """
    hm_height, hm_width = hm_shape
    hm_height = hm_height * 4
    hm_width = hm_width * 4

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])
            if output_3d:
                preds[i, j, 2] = coords[i, j, 2]

    return preds, pred_scores


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


class get_coord(object):
    def __init__(self, cfg, norm_size, output_3d=False):
        self.type = cfg.TEST.get('HEATMAP2COORD')
        self.input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self.norm_size = norm_size
        self.output_3d = output_3d

    def __call__(self, output, bbox, idx):
        if self.type == 'coord':
            pred_jts = output.pred_jts[idx]
            pred_scores = output.maxvals[idx]
            return heatmap_to_coord(pred_jts, pred_scores, self.norm_size, bbox, self.output_3d)
        # elif self.type == 'heatmap':
        #     pred_hms = output.heatmap[idx]
        #     return heatmap_to_coord_simple(pred_hms, bbox)
        else:
            raise NotImplementedError


# align for the test
def align_w_scale(mtx1: np.ndarray, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return mtx2_t, R, s, s1, s2, t1, t2
    else:
        return mtx2_t


def uvd2xyz(uvd, K, joint = 21):
    """
    checked
    """
    fx,fy,u0,v0 = K[:,0,0].view(-1,1,1), K[:,1,1].view(-1,1,1), K[:,0,2].view(-1,1,1), K[:,1,2].view(-1,1,1)
    u,v,z = uvd[:, :, 0].view(-1,joint,1), uvd[:, :, 1].view(-1,joint,1), uvd[:, :, 2].view(-1,joint,1)
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    xyz = torch.cat((x,y,z),-1)
    return xyz


def calculate_original_position_pytorch(preds, crop_center, crop_size, hand_side, resized_size):
    """
    checked
    """
    new_preds = preds.clone()
    bs = preds.shape[0]

    hand_side = hand_side.view(bs,-1)

    for i in range(bs):
        if hand_side[i]:
            new_preds[i, :, 0] = resized_size - new_preds[i, :, 0]

    current_center = crop_center.view(bs,1,2).to(preds.device)
    current_scale = (2*crop_size/resized_size).view(bs,1,1).to(preds.device)

    new_preds = (new_preds - resized_size/2) * current_scale + current_center
    return new_preds


def crop2xyz(uv_crop, norm_depth_pose, target, resized_size, camera_type = 'perspective', uv_norm=False):
    """
    checked
    """
    device = uv_crop.device
    if camera_type == 'perspective':
        crop_center, crop_size, hand_side, bone_length, pose3d_root, camera, rot_mat_inv = \
            target['crop_center'].to(device), target['crop_size'].to(device), target['hand_side'].to(device),\
            target['bone_length'].to(device),target['pose3d_root'].to(device), target['camera'].to(device), \
            target['rot_mat_inv'].to(device)
        if len(hand_side.shape) == 2:
            hand_side = hand_side[:, 0] == 1.

        if uv_norm:
            uv_crop = (uv_crop + 1) / 2 * 256

        pose2d, norm_depth_pose = uv_crop.clone().view(-1, 21, 2), norm_depth_pose.clone().view(-1, 21, 1)

        # rotate to original direction
        uv_aug_reverse = torch.ones(pose2d.shape[0], pose2d.shape[1], pose2d.shape[2] + 1).to(pose2d.device)
        uv_aug_reverse[:, :, :2] = pose2d
        uv_aug_reverse = uv_aug_reverse.matmul(rot_mat_inv)
        
        # translate and scale to original
        uv_original = calculate_original_position_pytorch(uv_aug_reverse, crop_center, crop_size, hand_side, resized_size)
        predicted_depth_pose = norm_depth_pose * bone_length.view(-1, 1, 1) + pose3d_root[:, 2].view(-1, 1, 1)
        predicted_depth_pose = predicted_depth_pose * 1000.0
        uvd = torch.cat((uv_original, predicted_depth_pose), -1)
        xyz = uvd2xyz(uvd, camera)/1000.0 # meter instead of mm

    elif camera_type == 'orthographic':
        crop_center, crop_size, hand_side, bone_length, pose3d_root = \
            target['crop_center'].to(device), target['crop_size'].to(device), target['hand_side'].to(device),\
            target['bone_length'].to(device),target['pose3d_root'].to(device)

        pose2d, norm_depth_pose = uv_crop.clone().view(-1, 21, 2), norm_depth_pose.clone().view(-1, 21, 1)
        uv_original = calculate_original_position_pytorch(pose2d, crop_center, crop_size, hand_side, resized_size)
        predicted_depth_pose = norm_depth_pose * bone_length.view(-1, 1, 1) + pose3d_root[:, 2].view(-1, 1, 1)
        uvd = torch.cat((uv_original, predicted_depth_pose), -1)
        xyz_normed,_ = batch_normalize_pose3d(uvd)

        for i in range(xyz_normed.shape[0]):
            if hand_side[i]:
                xyz_normed[i, :, 0] = -xyz_normed[i, :, 0]
        xyz = xyz_normed.clone() # return pose 3d normed

    return uv_original, xyz.float()
