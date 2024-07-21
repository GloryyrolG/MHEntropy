import torch


def target_transform(data: tuple, dataset_name: str) -> dict:
    """ Transforms to match the original RHD version

    Args:
        target:
            - coco: keys: target_uv
    """
    tsfm_target = {}
    if dataset_name in ['rhd', 'freihand', 'ho3d', 'mixed_ho3d_rhd']:
        image, tsfm_target = data
        tsfm_target['target_uvd_weight'] = torch.ones_like(tsfm_target['pose3d'])
    elif dataset_name == 'coco':
        image, target, _, bboxes = data
        tsfm_target['crop_uv'] = target['target_uv']
        tsfm_target['target_uv_weight'] = target['target_uv_weight']
    elif dataset_name == 'human3.6m':
        image, target, _, bboxes = data
        B = image.shape[0]
        tsfm_target['pose3d'] = target['target_xyz']
        tsfm_target['target_uvd_weight'] = target['target_uvd_weight']
        tsfm_target['scale'] = torch.ones(
            tsfm_target['pose3d'].shape[0], device=tsfm_target['pose3d'].device) 
        tsfm_target['crop_uv'] = target['target_uvd'].reshape(B, -1, 3)[..., : 2].flatten(-2)
        vis = target['target_uvd_weight'].reshape(B, -1, 3)[..., 0].clone()
        vis[vis == 0] = 2
        tsfm_target['vis'] = vis
        for k in ['st', 'st_cam', 'action']:
            tsfm_target[k] = target[k]
        tsfm_target['pose3d_root'] = target['root_xyz']
    else:
        raise NotImplementedError
    tsfm_target['image'] = image
    return image, tsfm_target
