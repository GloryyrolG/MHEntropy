import numpy as np
import cv2
import torch
from typing import Union

colorlist_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']


def plot_fingers(cloud, points, plt_specs, c, ax):
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(-4, 4)
    ax.set_zlim3d(-4, 4)
    ax.set_xlabel('x [unitScale]')
    ax.set_ylabel('y [unitScale]')
    ax.set_zlabel('z [unitScale]' )
    ax.axes.set_aspect('equal')

    # print(cloud)
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], zdir='z', s=2, c='r')
    for i in range(5):
        start, end = i*4+1, (i+1)*4+1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:,0], to_plot[:,1], to_plot[:,2], plt_specs, color=c[i])


def rot_x(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[1.0, 0.0, 0.0],
                       [0.0, cosAngle, -sinAngle],
                       [0.0, sinAngle, cosAngle]], dtype=float)
    return rotMat

def rot_y(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[cosAngle, 0.0, sinAngle],
                       [0.0, 1.0, 0.0],
                       [-sinAngle, 0.0, cosAngle]], dtype=float)
    return rotMat

def rot_z(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[cosAngle, -sinAngle, 0.0],
                      [sinAngle, cosAngle, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
    return rotMat

def viewCorrectionJoint(center_crop, cam_matrix, joint):
    f = cam_matrix[0,0]
    u0 = cam_matrix[0, 2]
    v0 = cam_matrix[1, 2]
    aroundYAngle = np.arctan((center_crop[0] - u0) / f)
    center3D = [center_crop[0] - u0, center_crop[1] - v0, f]
    center3DRotated = np.matmul(center3D, np.transpose(rot_y(-aroundYAngle)))
    aroundXAngle = np.arctan((center3DRotated[1]) / center3DRotated[2])
    viewRotation = np.matmul(rot_x(aroundXAngle), rot_y(-aroundYAngle))
    joint = np.matmul(joint, np.transpose(viewRotation))
    return viewRotation, joint

def viewCorrection(center_crop, cam_matrix, cloud, joint):
    # here for RHD, fx = fy and u0 = v0
    f = cam_matrix[0,0]
    u0 = cam_matrix[0,2]

    aroundYAngle = np.arctan((center_crop[0]-u0)/f)
    center3D = [center_crop[0]-u0, center_crop[1]-u0, f]
    center3DRotated = np.matmul(center3D, np.transpose(rot_y(-aroundYAngle)))
    aroundXAngle = np.arctan((center3DRotated[1])/center3DRotated[2])

    viewRotation = np.matmul(rot_x(aroundXAngle), rot_y(-aroundYAngle))
    cloud = np.matmul(cloud, np.transpose(viewRotation))
    joint = np.matmul(joint, np.transpose(viewRotation))

    return viewRotation, cloud, joint


def imcrop(img, center, crop_size):
    x1 = int(np.round(center[0]-crop_size))
    y1 = int(np.round(center[1]-crop_size))
    x2 = int(np.round(center[0]+crop_size))
    y2 = int(np.round(center[1]+crop_size))

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
         img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]

    return img_crop

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    if img.ndim < 3: # for depth
        borderValue = [0]
    else: # for rgb
        borderValue = [127, 127, 127]

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]],
                                     [0, 1, L - mid[1]]])

    img_list_out = []
    for img in img_list:
        img_list_out.append(cv2.warpAffine(img, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, label2d_list_out, camera


def uvd2xyz(uvd, cam_matrix):
    raise NotImplementedError
    f = cam_matrix[0,0]
    u0 = cam_matrix[0,2]
    x = (uvd[:, 0] - u0) * uvd[:, 2]/f
    y = (uvd[:, 1] - u0) * uvd[:, 2]/f
    z = uvd[:, 2]
    cloud = np.transpose(np.stack([x,y,z]))
    return cloud


def xyz2uvd(xyz, K) -> Union[np.ndarray, torch.Tensor]:
    if len(xyz.shape) == 2:
        fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        uvd = np.zeros_like(xyz, np.float32)
        uvd[:, 0] = (xyz[:, 0] * fx / (xyz[:, 2] + 1e-16) + fu)
        uvd[:, 1] = (xyz[:, 1] * fy / (xyz[:, 2] + 1e-16) + fv)
        uvd[:, 2] = xyz[:, 2]
        return uvd
    elif len(xyz.shape) == 3:
        # Batch
        bs = xyz.shape[0]
        paras = K
        fx, fy, fu, fv = paras[:,0,0].view(bs,1),paras[:,1,1].view(bs,1),\
        paras[:,0,2].view(bs,1),paras[:,1,2].view(bs,1)
        uvd = torch.rand_like(xyz).to(xyz.device)
        uvd[:, :, 0] = (xyz[:,:,0].view(bs,21) * fx / xyz[:,:,2].view(bs,21) + fu)
        uvd[:, :, 1] = (xyz[:,:,1].view(bs,21) * fy / xyz[:,:,2].view(bs,21) + fv)
        uvd[:, :, 2] = xyz[:,:,2]
        return uvd
    else:
        raise NotImplementedError
    

def depth2cloud(depth, mask, center3d, cam_matrix, cloudSize=5000):
    u = np.tile(np.arange(320), 320)
    v = np.repeat(np.arange(320), 320)
    u = np.reshape(u,[320*320])
    v = np.reshape(v,[320*320])
    mask = np.reshape(mask, [320*320])
    d =  np.reshape(depth, [320*320])
    u = u[mask]
    v = v[mask]
    d = d[mask]

    uvd = np.transpose(np.stack([u.astype(float),v.astype(float),d]))
    cloud = uvd2xyz(uvd, cam_matrix)

    cloud_rel = cloud - center3d
    boundingBoxSize=0.2
    validIndicies = np.logical_and(np.logical_and(np.abs(cloud_rel[:,0])<boundingBoxSize, np.abs(cloud_rel[:,1])<boundingBoxSize),
                                   np.abs(cloud_rel[:,2])<boundingBoxSize)
    cloud = cloud[validIndicies, :]
    if len(cloud) == 0:
        cloud = np.zeros((cloudSize, 3), dtype=np.float32)
    while len(cloud) < cloudSize:
        cloud = np.repeat(cloud, 2, axis=0)

    # randInidices = np.arange(len(cloud))
    # np.random.seed(0)
    # np.random.shuffle(randInidices)
    import torch
    randInidices=torch.randperm(len(cloud)).numpy()
    #print("random",randInidices[:3])
    cloudSampled = cloud[randInidices[0:cloudSize, ], :]

    return cloudSampled

def creat_2Dheatmap(kp_coord_uv, crop_center, crop_size, crop_scale,  kp_visible, sigma = 25.0):
    keypoint_uv21_u = (kp_coord_uv[:, 0] - crop_center[0]) * crop_scale + crop_size // 2
    keypoint_uv21_v = (kp_coord_uv[:, 1] - crop_center[1]) * crop_scale + crop_size // 2
    coords_uv = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
    coords_uv = np.stack([coords_uv[:, 1], coords_uv[:, 0]], -1)

    s = coords_uv.shape
    coords_uv = coords_uv.astype(np.int32)

    valid_vec = kp_visible.astype(np.float32)
    valid_vec = np.squeeze(valid_vec)
    cond_val = np.greater(valid_vec, 0.5)

    cond_1_in = np.logical_and(np.less(coords_uv[:, 0], crop_size - 1), np.greater(coords_uv[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_uv[:, 1], crop_size - 1), np.greater(coords_uv[:, 1], 0))
    cond_in = np.logical_and(cond_1_in, cond_2_in)
    cond = np.logical_and(cond_val, cond_in)

    coords_uv = coords_uv.astype(np.float32)

    # create meshgrid
    x_range = np.expand_dims(np.arange(crop_size), 1)
    y_range = np.expand_dims(np.arange(crop_size), 0)

    X = np.tile(x_range, [1, crop_size]).astype(np.float32)
    Y = np.tile(y_range, [crop_size, 1]).astype(np.float32)

    X = np.reshape(X, (crop_size, crop_size))
    Y = np.reshape(Y, (crop_size, crop_size))

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    X_b = np.tile(X, [1, 1, s[0]])
    Y_b = np.tile(Y, [1, 1, s[0]])

    X_b -= coords_uv[:, 0]
    Y_b -= coords_uv[:, 1]

    dist = np.square(X_b) + np.square(Y_b)

    heatmap =  np.exp(-dist / np.square(sigma)) * cond.astype(np.float32)

    return heatmap

def creat_refine_uv(kp_coord_uv, crop_center, crop_size, crop_scale):
    keypoint_uv21_u = (kp_coord_uv[:, 0] - crop_center[0]) * crop_scale + crop_size // 2
    keypoint_uv21_v = (kp_coord_uv[:, 1] - crop_center[1]) * crop_scale + crop_size // 2
    coords_uv = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
    return coords_uv


def preprocessSample(image, depth, mask, kp_coord_uv, kp_visible, kp_coord_xyz, camera_intrinsic_matrix):
    cond_l = np.logical_and(mask>1, mask<18)
    cond_r = mask>17
    num_px_left_hand = np.sum(cond_l)
    num_px_right_hand = np.sum(cond_r)
    hand_side = num_px_left_hand>num_px_right_hand
    if hand_side:
        pose3d = kp_coord_xyz[:21, :]
    else:
        pose3d = kp_coord_xyz[-21:, :]

    if hand_side:
        maskSingleHand = cond_l
    else:
        maskSingleHand = cond_r

    pose3d_root = pose3d[12, :]  # this is the root coord
    pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
    index_root_bone_length = np.sqrt(np.sum(np.square(pose3d_rel[12, :] - pose3d_rel[11, :])))
    scale = index_root_bone_length
    pose3d_normed = pose3d_rel / scale

    # crop image
    if hand_side:
        pose_uv_all = kp_coord_uv[:21, :]
        uv_vis = kp_visible[:21]
    else:
        pose_uv_all = kp_coord_uv[-21:, :]
        uv_vis = kp_visible[-21:]

    # get corresponding point cloud
    cloud = depth2cloud(depth, maskSingleHand, pose3d_root, camera_intrinsic_matrix, 4000)
    cloud_rel = cloud - pose3d[12, :]
    cloud_normed = cloud_rel / scale

    crop_center = pose_uv_all[12, :]
    crop_center = np.reshape(crop_center,2)

    pose_uv_vis = pose_uv_all[uv_vis, :]
    crop_size = np.max(np.absolute(pose_uv_vis-crop_center))*1.3
    # crop_size = np.minimum(np.maximum(crop_size, 25.0), 200.0)

    # Normalize depth maps
    # set the depth of palm_root as zero
    depth = (5.0 - depth) / 5.0  # set the range 0.0 - 1.0 ie. the farthest is 0.0
    depth = np.multiply(depth, maskSingleHand) # set the background 0.0

    # print crop_size
    image_crop = imcrop(image, crop_center, crop_size)
    depth_crop = imcrop(depth, crop_center, crop_size)
    mask_crop = imcrop(maskSingleHand.astype(np.float), crop_center, crop_size)

    image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
    depth_crop = cv2.resize(depth_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_crop = cv2.resize(mask_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

    # Calc s, t of uv, from crop and resize.
    t = np.array([image.shape[0] / 2., image.shape[1] / 2.]) - crop_center
    s = 256 / (2. * crop_size)

    # view correction
    viewRotation, cloud_vc, pose3d_vc = viewCorrection(crop_center, camera_intrinsic_matrix, cloud, pose3d) #vc== view corrected
    cloud_vc_rel = cloud_vc - pose3d_vc[12, :]
    cloud_vc_normed = cloud_vc_rel / scale
    pose3d_vc_rel = pose3d_vc - pose3d_vc[12, :]  # relative coords in metric coords
    pose3d_vc_normed = pose3d_vc_rel / scale

    if hand_side:
        hand_side_out = np.array([1.0, 0.0])
    else:
        hand_side_out = np.array([0.0, 1.0])

    heatmap_size = 64
    crop_scale = heatmap_size / (crop_size * 2)
    heatmap = creat_2Dheatmap(pose_uv_all, crop_center, heatmap_size, crop_scale, uv_vis, sigma=6.0)

    refine_size = 256
    crop_scale = refine_size / (crop_size * 2)
    crop_uv = creat_refine_uv(pose_uv_all, crop_center, refine_size, crop_scale)

    return image_crop, np.float32(depth_crop), np.float32(cloud_normed),  np.float32(pose3d_normed),\
           np.float32(cloud_vc_normed),np.float32(pose3d_vc_normed),  np.float32(viewRotation),\
           np.float32(scale), np.float32(hand_side_out), np.float32(heatmap),\
           (crop_center, crop_size, pose3d_root), (s, t), np.float32(crop_uv),\
           uv_vis, mask_crop