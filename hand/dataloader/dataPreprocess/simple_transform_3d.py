# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import math
import random

import os
import cv2
import numpy as np
import torch
import torchvision.utils as vutils

from dataloader.dataPreprocess.bbox import _box_to_center_scale, _center_scale_to_box
from dataloader.dataPreprocess.transforms import (affine_transform, flip_joints_3d,
                          get_affine_transform, im_to_torch)

from dataloader.dataPreprocess.pose_utils import cam2pixel, pixel2cam
from dataloader.rhddataloader import compute_st
from dataloader.Freihand_dataloader_3d import rotate


class SimpleTransform3D(object):
    """Generation of cropped input person and pose heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion, add_dpg,
                 input_size, output_size, bbox_3d_shape,
                 rot, sigma, train, loss_type='heatmap', scale_mult=1.25):
        self._dataset = dataset
        self._joint_pairs = dataset.joint_pairs
        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        self.skeleton = dataset.skeleton

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        return img, bbox

    def _target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        assert np.all(joints_3d[:, 0, 1] == 1)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5  # [-0.5, 0.5)
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[0]  # 2000

        # Out-of-range.
        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0  # outlier?
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        bbox = list(label['bbox'])
        joint_img = label['joint_img'].copy()
        joints_vis = label['joint_vis'].copy()
        root_cam = label['root_cam'].copy()
        self.num_joints = joint_img.shape[0]

        gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
        gt_joints[:, :, 0] = joint_img
        gt_joints[:, :, 1] = joints_vis

        imgwidth, imght = label['width'], label['height']
        assert imgwidth == src.shape[1] and imght == src.shape[0]
        # self.num_joints = gt_joints.shape[0]

        input_size = self._input_size

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        # xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

        #TODO: half body transform
        if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
            raise ValueError
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints[:, :, 0], joints_vis
            )

            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body

        # rescale. Scale crop.
        if self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0

        if self._train and self._occlusion:
            raise ValueError
            while True:
                area_min = 0.0
                area_max = 0.7
                synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                ratio_min = 0.3
                ratio_max = 1 / 0.3
                synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                synth_h = math.sqrt(synth_area * synth_ratio)
                synth_w = math.sqrt(synth_area / synth_ratio)
                synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                    synth_xmin = int(synth_xmin)
                    synth_ymin = int(synth_ymin)
                    synth_w = int(synth_w)
                    synth_h = int(synth_h)
                    src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                    break

        joints = gt_joints
        flipped = False
        if random.random() > 0.5 and self._train:
            flipped = True
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])  # no t
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # Only R aug around the bbox center.
        abs_25d = joints[..., 0].copy()
        th = -r / 180 * np.pi
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        abs_25d[:, : 2] = center + (abs_25d[:, : 2] - center) @ R.T
        # Ignore the change of d.

        # s, R, t.
        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # 2nd unconventional mthd is to store the rel dep.

        # generate training targets
        if self._loss_type == 'heatmap':
            target, target_weight = self._target_generator(joints, self.num_joints)
        elif self._loss_type == 'coord':
            target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w)
        target_weight *= joints_vis.reshape(-1)
        bbox = _center_scale_to_box(center, scale)

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)

        # vutils.save_image(img[None, ...], os.path.join('tmp', 'human3.6m.png'))

        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

		### Added. (to revise)
        target = target * 2  # [-0.5, 0.5) -> [-1, 1); / 2000 -> / 1000

        # No scale norm.
        # abs_25d = joint_img.copy()
        abs_25d[:, 2] = abs_25d[:, 2] + root_cam[2]

        # bbox_c = center
        # if flipped:
        #     jp_np = np.array(self._joint_pairs)
        #     # rel_3d[:, 0] = -rel_3d[:, 0]
        #     # Left and right part!
        #     # rel_3d[jp_np[:, 0]], rel_3d[jp_np[:, 1]] = (rel_3d[jp_np[:, 1]],
        #     #                                             rel_3d[jp_np[:, 0]])
        #     abs_25d[:, 0] = imgwidth - 1 - abs_25d[:, 0]
        #     abs_25d[jp_np[:, 0]], abs_25d[jp_np[:, 1]] = (abs_25d[jp_np[:, 1]],
        #                                                   abs_25d[jp_np[:, 0]])
        
        # flip, affine (s, R, t), norm.
        abs_3d = pixel2cam(abs_25d, label['f'], label['c'])
        root_idx = self._dataset.root_idx
        rel_3d = abs_3d - abs_3d[root_idx]
        rel_3d = rel_3d / 1000

        rel_3d = rel_3d.reshape(-1)
        abs_25d = abs_25d.reshape(-1)
        uv = target.reshape(-1, 3)[:, : 2].reshape(-1)
        # Orth proj.
        st =     compute_st(rel_3d,  uv, root_idx=root_idx)

        st_cam = compute_st(abs_25d, uv, root_idx=root_idx)
        K = np.array([
            [label['f'][0], 0, label['c'][0]],
            [0, label['f'][1], label['c'][1]],
            [0,             0,             1]], dtype=np.float32)
        A = np.array([
            [st_cam[0], 0, st_cam[1]],
            [0, st_cam[0], st_cam[2]],
            [0,         0,         1]], dtype=np.float32)
        C = A @ K
        st_cam = np.concatenate((C[[0], 0], C[[1], 1], C[: 2, 2]))

        ###

        output = {
            'type': '3d_keypoint_data',
            'image': img,
            'target_uvd': torch.from_numpy(target).float(),
            'target_uvd_weight': torch.from_numpy(target_weight).float(),
            'bbox': torch.Tensor(bbox),
            'target_xyz': torch.from_numpy(rel_3d).float(),
            'st': torch.from_numpy(st).float(),
            'st_cam': torch.from_numpy(st_cam),
            'root_xyz': torch.from_numpy(abs_3d[root_idx] / 1000).float(),
            'action': label['action']
        }
        return output

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
