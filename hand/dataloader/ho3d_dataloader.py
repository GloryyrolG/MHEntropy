from os.path import join
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import torchvision
from torchvision import transforms
import copy
import math
import cv2,os
import imageio
import random
import numpy as np
from dataloader.ho3d_vis_utils import _assert_exist,read_annotation,read_obj,read_depth_img
from dataloader.rhddataloader import compute_st

ho3d2RHD_skeidx = [0,16,15,14,13,17,3,2,1,18,6,5,4,19,12,11,10,20,9,8,7]

###############################Set path#############################################

data_root = './datasets/HO3D_v3/HO3D_v3/'
ycb_root =  './datasets/HO3D_v3/models/'
gt_root = './datasets/HO3D_v3/HO3D/data/'
seg_root = './datasets/HO3D_v3/'

_assert_exist(data_root)
_assert_exist(ycb_root)
_assert_exist(gt_root)
_assert_exist(seg_root)

###############################HO3D dataset process#############################################
def coord_change(xyz):
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    xyz = xyz.dot(coord_change_mat.T)
    return xyz


def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2], proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 3

    return proj_pts

def uvd2xyz(uvd, K):
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]

    xyz = xyz.dot(coord_change_mat.T)
    return xyz

def xyz2uvd(xyz, K):
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    xyz = xyz.dot(coord_change_mat.T)
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def get_bbox_joints(joints2d, bbox_factor=1.1):
    min_x, min_y = joints2d.min(0)
    max_x, max_y = joints2d.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    bbox_delta_x = (max_x - min_x) * bbox_factor / 2
    bbox_delta_y = (max_y - min_y) * bbox_factor / 2
    bbox_delta = np.asarray([bbox_delta_x, bbox_delta_y])
    bbox = np.array([*(center - bbox_delta), *(center + bbox_delta)], dtype=np.float32)
    return bbox

def fuse_bbox(bbox_1, bbox_2, img_shape, scale_factor=1.):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    #print(max_delta)
    scale = max_delta * scale_factor
    return center, scale

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

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def processing_pose3d(pose3d, root_index=4, relative_index=5):
    pose3d_root = pose3d[root_index, :]  # this is the root coord
    pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
    index_root_bone_length = np.sqrt(np.sum(np.square(pose3d_rel[root_index, :] - pose3d_rel[relative_index, :])))
    scale = index_root_bone_length
    pose3d_normed = pose3d_rel / scale
    return pose3d_rel, pose3d_normed, pose3d_root, index_root_bone_length

def processing_augmentation(image, pose3d, obj_mask, mask, depth, uv):  #scale, transiltion， uv argumentation is different from the xyz vertices
    randScaleImage = np.random.uniform(low=0.8, high=1.0)
    pose3d_aug = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0] #change the rotation to
    # randAngle = 0.
    rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
                                     randScaleImage)  # change image later together with translation

    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY


    (pose3d_aug[:, 0], pose3d_aug[:, 1]) = rotate((0, 0), (pose3d_aug[:, 0], pose3d_aug[:, 1]), randAngle)
    uv_aug = np.ones([uv.shape[0], uv.shape[1] + 1])
    uv_aug[:, :2] = uv
    uv_aug = np.dot(rotMat, uv_aug.T).T

    image_aug = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    mask_aug = cv2.warpAffine(mask.astype(np.float32), rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    obj_mask_aug = cv2.warpAffine(obj_mask.astype(np.float32), rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    depth_aug = cv2.warpAffine(depth, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    image_aug = np.reshape(image_aug, [256, 256, 3])
    mask_aug = np.reshape(mask_aug, [256, 256]).astype(bool)
    obj_mask_aug = np.reshape(obj_mask_aug, [256, 256]).astype(bool)
    depth_aug = np.reshape(depth_aug, [256, 256])
    return image_aug, pose3d_aug, obj_mask_aug, mask_aug, depth_aug, uv_aug, rotMat

def rgb_processing(rgb_img):
    # in the rgb image we add pixel noise in a channel-wise manner
    noise_factor = 0.4
    pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    return rgb_img

class Generate_ho3d_uv(data.Dataset):
    def __init__(
            self, joint_idx = 'RHD', data_root=data_root, ycb_root=ycb_root,
            gt_root=gt_root, mode='training', dpda='HO3D'):
        super(Generate_ho3d_uv, self).__init__()

        # load annotations
        self.baseDir = data_root
        self.model = "train" #evaluation
        if mode not in ['training', 'evaluation']:
            raise ValueError
        self.mode = mode
        self.aug = self.mode == 'training'
        self.joint_idx = joint_idx
        assert self.joint_idx in ['RHD','Ho3D'], 'Joint Index Error'
        self.dpda = dpda

        train_file = []
        f_train = open(self.baseDir + 'train.txt')
        line = f_train.readline()
        train_file.append(line.strip('\n'))
        while line:
            line =f_train.readline()
            train_file.append(line.strip('\n'))
        self.train_file = np.array(train_file)[:-1]
        print(self.train_file.shape)
        print("training data:", self.train_file.shape[0])
        print(f"self.train_file[0] {self.train_file[0]}")

        handjoint3d_4w = np.load(gt_root+"handJoints3D_train_4w.npy")
        print(f"handjoint3d_4w.shape {handjoint3d_4w.shape}")
        handjoint3d_8w = np.load(gt_root+"handJoints3D_train_8w.npy")
        print(f"len(handjoint3d_8w) {len(handjoint3d_8w)}")
        handjoint3d_left = np.load(gt_root+"handJoints3D_train_left.npy")
        ho3d_mesh_train_4w = np.load(gt_root+"ho3d_mesh_train_4w.npy")
        ho3d_mesh_train_8w = np.load(gt_root+"ho3d_mesh_train_8w.npy")
        ho3d_mesh_train_left = np.load(gt_root+"ho3d_mesh_train_left.npy")

        self.handJoints3D = np.concatenate((handjoint3d_4w, handjoint3d_8w, handjoint3d_left))
        print(f"len(self.handJoints3D) {len(self.handJoints3D)}")
        self.handMesh = np.concatenate((ho3d_mesh_train_4w, ho3d_mesh_train_8w, ho3d_mesh_train_left))

        ### Train val split.
        eval_seqNames = ['ABF14', 'MC5', 'SB14', 'ShSu13']  # , 'SMu41']
        # eval_seqNames = ['ShSu13']
        # print("> For viz!")
        indices = []
        for idx, s in enumerate(self.train_file):
            seqName = s.split('/')[0]
            if (self.mode == 'training' and seqName in eval_seqNames
                    or self.mode == 'evaluation' and seqName not in eval_seqNames):
                continue
            indices.append(idx)
        self.train_file = self.train_file[indices]
        self.handJoints3D = self.handJoints3D[indices]
        self.handMesh = self.handMesh[indices]

        #########
        
        objName_list = os.listdir(os.path.join(ycb_root))
        if '.DS_Store' in objName_list: objName_list.remove('.DS_Store')

        objmesh_all = {}
        for i in range(len(objName_list)):
            objMesh = read_obj(os.path.join(ycb_root, objName_list[i], 'textured_simple.obj'))
            objmesh_all[objName_list[i]] = objMesh
        self.objmesh_all = objmesh_all

    def __len__(self):
        return len(self.train_file)
        #return 100

    def __getitem__(self, idx):
        # idx = 1690
        # print(idx)
        seqName = self.train_file[idx].split('/')[0]
        id = self.train_file[idx].split('/')[1]

        img_filename = os.path.join(self.baseDir, self.model, seqName, 'rgb', id + '.jpg')
        seg_filename = os.path.join(seg_root, self.model, seqName, 'seg', id + '.png')
        image = imageio.imread(img_filename)  # image size 480 x 640
        depth = read_depth_img(self.baseDir, seqName, id, self.model)
        depth = np.array(depth)


        seg = imageio.imread(seg_filename)   # seg size 120 x 160
        seg = cv2.resize(seg, (640, 480), interpolation=cv2.INTER_NEAREST)
        anno = read_annotation(self.baseDir, seqName, id, self.model)
        # get the hand Mesh from MANO model for the current pose
        handJoints3D = self.handJoints3D[idx]
        handMesh = self.handMesh[idx]
        objMesh = self.objmesh_all[anno['objName']]

        # # apply current pose to the object model
        objMesh_new = np.matmul(objMesh['v'], cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
        objMeshNorm_new = np.matmul(objMesh['vn'], cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

        # # project to 2D generate the uvd coordinates
        handJoints3D = handJoints3D*1000.0 #transfer to the mm depth is negative,
        handMesh = handMesh*1000.0
        objMesh_new = objMesh_new*1000.0
        objMeshNorm_new = objMeshNorm_new * 1000.

        handJoints_uvd = xyz2uvd(handJoints3D, anno['camMat'])
        objMesh_uvd = xyz2uvd(objMesh_new, anno['camMat'])
        handMesh_uvd = xyz2uvd(handMesh, anno['camMat'])

        # Coord change.
        handJoints3D = coord_change(handJoints3D)
        handMesh = coord_change(handMesh)
        objMesh_new = coord_change(objMesh_new)
        objMeshNorm_new = coord_change(objMeshNorm_new)

        handJoints2d = handJoints_uvd[:,:2]
        objMesh2d = objMesh_uvd[:,:2]
        handMesh2d = handMesh_uvd[:,:2]

        bbox_hand = get_bbox_joints(handJoints2d, bbox_factor=1.5)
        crop_obj = get_bbox_joints(objMesh2d, bbox_factor=1.0)

        if self.dpda == 'HO3D': 
            # Hand crop padded with 0.
            # center = handJoints2d[ho3d2RHD_skeidx[12]]
            # scale = np.max(np.absolute(handJoints2d - center)) * 1.3 * 2
            # h, w = image.shape[: 2]
            # new_image = np.zeros_like(image)
            # xmin, xmax = max(0, int(center[0] - scale / 2)), min(w, int(center[0] + scale / 2))
            # ymin, ymax = max(0, int(center[1] - scale / 2)), min(h, int(center[1] + scale / 2))
            # new_image[ymin: ymax, xmin: xmax] = image[ymin: ymax, xmin: xmax]
            # image = new_image

            center, scale = fuse_bbox(bbox_hand, crop_obj, image.shape)
        elif self.dpda == 'RHD':
            # Consistent w/ RHD.
            center = handJoints2d[ho3d2RHD_skeidx[12]]
            scale = np.max(np.absolute(handJoints2d - center)) * 1.3 * 2

        crop_center_rgb = center
        crop_size_rgb = scale/2

        image_crop = imcrop(image, crop_center_rgb, crop_size_rgb)
        image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        depth_crop = imcrop(depth, crop_center_rgb, crop_size_rgb)
        depth_crop = cv2.resize(depth_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        seg_crop = imcrop(seg, crop_center_rgb, crop_size_rgb)
        seg_crop = cv2.resize(seg_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        object_mask_crop = seg_crop[:,:,1] > 200
        hand_mask_crop = seg_crop[:,:,2] > 200
        
        hand_mask = seg[:,:,2] > 200

        pose_uvd_relatived = copy.deepcopy(handJoints_uvd)
        pose_uvd_relatived[:,2] = pose_uvd_relatived[:,2] - handJoints_uvd[0,2]
        pose_uvd_relatived_ortho = copy.deepcopy(pose_uvd_relatived)
        pose_uvd_relatived_ortho[:,0] = (pose_uvd_relatived_ortho[:,0] - crop_center_rgb.reshape(1,2)[0,0] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the x
        pose_uvd_relatived_ortho[:,1] = (pose_uvd_relatived_ortho[:,1] - crop_center_rgb.reshape(1,2)[0,1] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the y
        uv_crop = pose_uvd_relatived_ortho[:,:2]

        vis = np.zeros([21]).astype(bool)
        quant = 5
        for i in range(21):
            u,v,d = int(handJoints_uvd[i,0]),int(handJoints_uvd[i,1]),handJoints_uvd[i,2]
            u0, v0 = u, v
            flag = False
            for u in range(u0 - quant + 1, u0 + quant):
                for v in range(v0 - quant + 1, v0 + quant):
                    if ((u>=640) or (v>=480) or (v<0) or (u <0)): 
                        pass
                    elif (hand_mask[v, u]):
                        # print(d - depth[v,u]*1000)
                        if (d - depth[v,u]*1000) < 40:
                            flag = True
                            break
                if flag:
                    break
            vis[i] = flag  # set viz
        
        pose3d_rel, pose3d_normed, pose3d_root, index_root_bone_length = processing_pose3d(handJoints3D)


        rotMat = np.eye(2, 3)
        if self.aug:
            if self.dpda == 'HO3D':
                image_crop = rgb_processing(image_crop)
            image_crop, pose3d_normed, object_mask_crop, hand_mask_crop, depth_crop, uv_crop, rotMat = \
                processing_augmentation(image_crop, pose3d_normed, object_mask_crop, hand_mask_crop, depth_crop, uv_crop)
        
        for i in range(21):
            u,v = uv_crop[i,0],uv_crop[i,1]
            flag = False
            for du in range(-quant + 1, quant):
                for dv in range(-quant + 1, quant):
                    if ((u + du>255) or (v + dv>255) or (v + dv<0) or (u + du<0)):
                        pass
                    else:
                        flag = True
                        break
                if flag:
                    break
            if not flag:
                vis[i] = False  # viz --> inviz

        if self.dpda == 'HO3D':
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        elif self.dpda == 'RHD':
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            if self.aug:
                image_trans.transforms.insert(0, torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4,1.6], saturation=[0.4,1.6], hue=0.1))
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        if self.joint_idx == 'RHD':
            uv_crop = uv_crop[ho3d2RHD_skeidx,:]
            handJoints3D = handJoints3D[ho3d2RHD_skeidx,:]
            pose3d_normed = pose3d_normed[ho3d2RHD_skeidx,:]
            vis = vis[ho3d2RHD_skeidx]  # not forget to rearrange as well
        uv_crop = uv_crop / 256 * 2 - 1

        # get rot_mat_inv
        rot_mat_inv = np.eye(3)
        rot_mat_inv[:2, :] = rotMat
        rot_mat_inv = np.linalg.inv(rot_mat_inv.T)
        rot_mat_inv = rot_mat_inv[:, :2]

        st = compute_st(pose3d_normed, uv_crop)

        target = {}
        target['crop_uv'] =  torch.from_numpy(uv_crop).float().flatten(-2)
        target['hand_mask'] = torch.from_numpy(hand_mask_crop).bool()
        target['object_mask'] = torch.from_numpy(object_mask_crop).bool()
        target['vis'] = torch.from_numpy(vis).float()
        target['depth'] = torch.from_numpy(depth_crop).float()
        target['original_pose3d'] = torch.from_numpy(handJoints3D).float()
        target['verts'] = torch.from_numpy(handMesh).float().flatten(-2)
        target['pose3d'] = torch.from_numpy(pose3d_normed).float().flatten(-2)
        target['pose3d_root'] = torch.from_numpy(pose3d_root).float()  # raw
        target['_root_idx'] = 12
        target['st'] = torch.from_numpy(st.astype(np.float32))
        target['patch'] = np.zeros(3, dtype=np.float32)
        target['scale'] = torch.tensor(index_root_bone_length / 1000.).float() # metric m
        target['object_verts'] = torch.from_numpy(objMesh_new[
            np.sort(np.random.choice(objMesh_new.shape[0], 1000, replace=False))]).float().flatten(-2)
        # target['object_faces'] = objMesh['f'].astype('int')
        # target['object_vertns'] = torch.from_numpy(objMeshNorm_new).float().flatten(-2)

        target['crop_center'] = torch.from_numpy(crop_center_rgb.astype(np.float32))
        target['crop_size'] = torch.tensor(crop_size_rgb).float()
        target['hand_side'] = torch.zeros([], dtype=torch.float32)
        target['bone_length'] = target['scale']
        target['pose3d_root'] = torch.from_numpy((handJoints3D[12] / 1000).astype(np.float32))
        target['camera'] = torch.from_numpy(anno['camMat']).float()
        target['rot_mat_inv'] = torch.from_numpy(rot_mat_inv.astype(np.float32))
        target['_rot_mat'] = torch.from_numpy((rotMat[:, : 2] / np.linalg.norm(rotMat[0, : 2])).astype(np.float32))
        target['uvd'] = torch.from_numpy(np.concatenate([uv_crop, pose3d_normed[:, [-1]]], axis=1).astype(np.float32).ravel())

        target['dataset'] = 'ho3d'
        target['idx'] = idx

        return  image_crop.type(torch.FloatTensor),target

if __name__ == '__main__':

    UV_Map_Generator_train = Generate_ho3d_uv(joint_idx='RHD', mode='evaluation')
    # data_loader_train = torch.utils.data.DataLoader(UV_Map_Generator_train,batch_size=1,shuffle=True, num_workers=1)

    # loader = iter(data_loader_train)
    image, target = UV_Map_Generator_train.__getitem__(3300)

    image_crop = image.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
    image_crop = (image_crop + 1)/2
    uv = target['crop_uv'].squeeze().detach().numpy()
    vis = target['vis'].squeeze().detach().numpy().astype(bool)
    print(vis)
    depth = target['depth'].squeeze().detach().numpy()
    seg_crop = target['hand_mask'].squeeze().detach().numpy()
    pose3d = (target['pose3d']).squeeze().detach().numpy()


    plt.subplot(2, 3, 1)
    plt.axis('off')
    plt.imshow(image_crop)
    print(f"uv[12] {uv[12]}")
    plt.scatter(uv[:, 0], uv[:, 1], s=5, c='r')

    plt.subplot(2, 3, 2)
    plt.axis('off')
    plt.imshow(seg_crop)
    plt.scatter(uv[:, 0], uv[:, 1], s=5, c='r')

    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.imshow(seg_crop)
    plt.scatter(uv[vis, 0], uv[vis, 1], s=5, c='r')

    plt.subplot(2, 3, 4)
    plt.axis('off')
    plt.imshow(depth)
    plt.scatter(uv[vis, 0], uv[vis, 1], s=5, c='r')
    plt.show()
    plt.savefig('tmp/plt.png')

    from viz import plot_pose2d, plot_pose3d
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(image_crop,alpha=0.6)
    plot_pose2d(ax1, uv, order='uv', draw_kp=True, linewidth='1', markersize = 5, dataset='RHD')
    ax1.axis('off')
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    plot_pose3d(pose3d, '.-', ax1, azim=90.0, elev=180.0)
    plt.show()
    plt.close()















