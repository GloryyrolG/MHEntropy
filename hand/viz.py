from typing import Union
import trimesh,torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from ManoLayer import ManoLayer 
from utils import xyz2crop
from utils import align_w_scale


Bighand2RHD_skeidx = [0, 8, 7, 6, 1, 11, 10, 9, 2, 14, 13, 12, 3, 17, 16, 15, 4, 20, 19, 18, 5]  # depth
RHD2FreiHand_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']

# # Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
# H36M_NAMES = ['']*32
# H36M_NAMES[0]  = 'Hip'  # Pelvis/Bottom Torso?
# H36M_NAMES[1]  = 'RHip'
# H36M_NAMES[2]  = 'RKnee'
# H36M_NAMES[3]  = 'RFoot'  # Ankle?
# H36M_NAMES[6]  = 'LHip'
# H36M_NAMES[7]  = 'LKnee'
# H36M_NAMES[8]  = 'LFoot'
# H36M_NAMES[12] = 'Spine'  # Torso?
# H36M_NAMES[13] = 'Thorax'  # Upper Torso?
# H36M_NAMES[14] = 'Neck/Nose'  # together?
# H36M_NAMES[15] = 'Head'
# H36M_NAMES[17] = 'LShoulder'
# H36M_NAMES[18] = 'LElbow'
# H36M_NAMES[19] = 'LWrist'
# H36M_NAMES[25] = 'RShoulder'
# H36M_NAMES[26] = 'RElbow'
# H36M_NAMES[27] = 'RWrist'  # Hand?
RLE2MDN = [
    0, 1, 2, 3,    0, 0, 4, 5,
    6, 0, 0, 0,    7, 17, 8, 10,
    0, 11, 12, 13, 0, 0, 0, 0,
    0, 14, 15, 16, 0, 0, 0, 0
]  # 
    # 'Right Ankle', # 0
    # 'Right Knee', # 1
    # 'Right Hip', # 2
    # 'Left Hip', # 3
    # 'Left Knee', # 4
    # 'Left Ankle', # 5
    # 'Right Wrist', # 6
    # 'Right Elbow', # 7
    # 'Right Shoulder', # 8
    # 'Left Shoulder', # 9
    # 'Left Elbow', # 10
    # 'Left Wrist', # 11
    # 'Neck (LSP)', # 12. H36M, MPI-INF-3DHP
    # 'Top of Head (LSP)', # 13  # MPI-INF-3DHP
    # 'Pelvis (MPII)', # 14. H36M, MPI-INF-3DHP
    # 'Thorax (MPII)', # 15
    # 'Spine (H36M)', # 16. MPI-INF-3DHP
    # 'Jaw (H36M)', # 17. MPI-INF-3DHP
    # 'Head (H36M)', # 18
    # 'Nose', # 19
    # 'Left Eye', # 20
    # 'Right Eye', # 21
    # 'Left Ear', # 22
    # 'Right Ear' # 23
RLE2SPIN = [
    3, 2, 1,    4, 5, 6,
    16, 15, 14, 11, 12, 13,
    8, 10, 0,   17, 7, 8,  # Top of Head -> Head; Jaw -> Neck 
    10, 9, 10,  10, 10, 10,
]


def export_mano_mesh(vertices: Union[np.ndarray, torch.Tensor], mano_faces, file_name = './model/test/mano.stl'):
    """
    Args:
        vertices: shape: (B, N, 3) or (N, 3).
    
    Examples:
        mano_faces = ManoLayer().mano_layer.th_faces.detach().cpu().numpy().copy()

        mano = ManoLayer()
        rand = torch.rand(10,512)
        out = mano(rand)
        export_mano_mesh(out['mesh'])
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    mano_faces = mano_faces.cpu().numpy()
    if vertices.ndim == 3:  # only show the first sample if input is Bx778x3
        vertices = vertices[0]
    assert (vertices.ndim == 2) and isinstance(vertices, np.ndarray), 'Vertices Input Error'

    mesh = trimesh.Trimesh(vertices=vertices, faces=mano_faces)
    mesh.export(file_name)


def viz_2djoints(
        images: torch.Tensor, crop_uv=None, pose3d=None, targets=None, idx=0,
        dataset='RHD', imgpth='tmp/viz_2djoints.png'):
    """
    Args:
        image: shape: (B, C, H, W). It must be on CPU.
        crop_uv: shape: ((N,) B, K, 2)
        targets (Optional[dict]):.
        pose3d: shape: (B, 3K) or (N, B, 3K).
    """
    plt.close()
    if images is not None:
        plt.imshow(images[idx].permute(1, 2, 0))

    if crop_uv is not None:
        if len(crop_uv.shape) == 3:
            crop_uv = crop_uv[None, ...]
        N = crop_uv.shape[0]
    else:  # pose3d is not None
        if len(pose3d.shape) == 2:
            pose3d = pose3d[None, ...]
        N = pose3d.shape[0]
    for i in range(N):
        if crop_uv is not None:
            uv = crop_uv[i, idx]
        else:
            pose3d_ = pose3d[i]
            crop_uv_, _ = xyz2crop(pose3d_.view(pose3d_.shape[0], -1, 3), targets)
            uv = crop_uv_[idx]

        plot_pose2d(plt.gca(), uv.numpy(), linewidth='3', markersize=7, dataset=dataset)

        for no, uv_ in enumerate(uv):
            plt.text(uv_[0], uv_[1], f'{no}')
    plt.savefig(imgpth)


def plot_pose2d(
    axis, coords_hw: np.ndarray, vis=None, color_fixed=None, linewidth='1',
    order='uv', draw_kp=True, markersize = 15, dataset='RHD'):
    """
    Plots a hand stick figure into a matplotlib figure. revised based on Freihand
    input idx: Bighand
    hand idx: Freihand
    example:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(hand_image,alpha=0.6)
        plot_pose2d(ax1, pose2d, order='uv', draw_kp=True, linewidth='3',color_fixed='dimgray')
        ax1.axis('off')
        plt.show()
        plt.close()
    """
    if dataset == 'RHD':
        # coords_hw = coords_hw[Bighand2RHD_skeidx,:]
        coords_hw = coords_hw[RHD2FreiHand_skeidx,:]
    elif dataset == 'FreiHand':
        pass
    else:
        raise NotImplementedError
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])

        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth, alpha = 0.6, zorder=1)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth, zorder=1)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            if color_fixed is None:
                axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :], markersize = markersize, zorder=1)
            else:
                axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=color_fixed, markersize = markersize, zorder=1)

    axis.set_zorder(0)


def viz_ddpm_sampling(model, num_points):
    from plotly import graph_objects as go

    samples = model.sample(1024, num_points)
    samples = samples.reshape(-1, 2).detach().cpu().numpy()
    fig = go.Figure(go.Histogram2d(x=samples[:, 0], y=samples[:, 1]))
    fig.write_image('model/test/ddpm_sampling.png')


def plot_pose3d(
        points, plt_specs, ax, c = colorlist_gt, azim=-90.0, elev=180.0,
        grid=False, dataset='RHD', s=1):
    """
    revised based on Cross VAE Hand
    input idx: bighand
    hand idx: RHD
    set azim to 0 or 45 to get other view
    example:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        plot_pose3d(pose3d, '.-', colorlist_gt, ax1, azim=90.0, elev=180.0)
        plt.show()
        plt.close()

    Note that we switch the y axis and z axis (ax.plot(to_plot[:, 0], to_plot[:, 2], to_plot[:, 1])
    because we need the rotation along with y axis
    """
    assert points.size == 21 * 3, "pose3d should have 63 entries, it has %d instead" % points.size
    if dataset == 'RHD':
        # points = points[Bighand2RHD_skeidx,:]
        # points = points[RHD2FreiHand_skeidx,:]
        pass
    else:
        raise NotImplementedError
    for i in range(5):
        start, end = i * 4 + 1, (i + 1) * 4 + 1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 2], to_plot[:, 1], plt_specs, color=c[i], linewidth=1.5 * s)
        for j in range(1):
           ax.text(points[start + j, 0], points[start + j, 2], points[start + j, 1], f'{start + j}')

    ax.view_init(azim=azim, elev=elev)

    xroot, yroot, zroot = points[12, 0], points[12, 2], points[12, 1]
    # RADIUS = 2.  # np.abs(points - points[12]).max()  # 0.12  # space around the subject
    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.dist = 7.5

    if grid:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.grid(True)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()

        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)


def export_pose3d_gif(points, file = 'rotation.gif', azim=90.0, elev=180.0,
                      plot_func=plot_pose3d, func_kwargs={}):
    if len(func_kwargs) == 0:
        func_kwargs = {'plt_specs': '.-', 'c': colorlist_gt, 'grid': False}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # init_func. ax.view_init in the func will not affect.
    plot_func(points, ax=ax, **func_kwargs)
    def rotate(angle):
        ax.view_init(azim=angle,elev=elev)
    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=(azim+np.arange(0, 362, 10))%360, init_func=None,
        interval=100)
    if file.endswith('.mp4'):
        writer = animation.FFMpegWriter()
    else:  # .gif
        writer = 'imagemagick'
    rot_animation.save(file, dpi=80, writer=writer)
    plt.close()


def export_pose3d_w_gt_gif(
        points, gt, file = 'rotation.gif', azim=90.0, elev=180.0,
        aligned=True, plot_func=plot_pose3d, func_kwargs={}, func_kwargs_1={}):
    """
    Args:
        points (np.ndarray): shape: (K, 3)
        gt (np.ndarray): shape: (K, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(func_kwargs) == 0:
        func_kwargs = {'plt_specs': '.-', 'c': colorlist_gt, 'grid': False}
        func_kwargs_1 = dict(func_kwargs)
        func_kwargs_1['c'] = ['#660004', '#CF0000', '#DE6669', '#EC9998',
                              '#E7B7B1']
    if aligned:
        points = align_w_scale(gt, points)
    plot_func(points, ax=ax, **func_kwargs)
    plot_func(gt,     ax=ax, **func_kwargs_1)
    def rotate(angle):
        ax.view_init(azim=angle,elev=elev)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=(azim+np.arange(0, 362, 10))%360, interval=100)
    if file.endswith('.mp4'):
        writer = animation.FFMpegWriter()
    else:  # .gif
        writer = 'imagemagick'
    rot_animation.save(file, dpi=60, writer=writer)
    plt.close()


def show3Dpose(channels, ax, cam=False, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False): # blue, orange
  """ (MDN)
  Visualize the ground truth 3d skeleton
  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(['']*32)*3, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(['']*32), -1) )
  if cam:
      vals = vals.copy()
      vals = vals[:, [2, 0, 1]]

  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    # ax.plot(x, y, z, lw=3, marker = 'o', markersize = 5, c=lcolor if LR[i] else rcolor, markeredgecolor = lcolor)
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  # ax.set_xlim3d([np.min(vals[:, 0]), np.max(vals[:, 0])])
  # ax.set_zlim3d([np.min(vals[:, 2]), np.max(vals[:, 2])])
  # ax.set_ylim3d([np.min(vals[:, 1]), np.max(vals[:, 1])])

  if add_labels:
    if cam:
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
  else:
    # Get rid of the ticks and tick labels

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])
  # ax.set_aspect('equal')
  ax.set_box_aspect([1,1,1])

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  # ax.w_zaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # # Keep z pane
  #
  # # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


  ax.view_init(azim=129, elev=10)


def mesh_axis_tsfm(verts):
    """
    Args:
        verts: shape: (N, B, V)
    """
    # Axis tsfm for mesh viz.
    verts = verts.clone().reshape(*verts.shape[: -1], -1, 3)  # remember to clone first for in-place ops
    verts[..., 1] *= -1
    verts[..., 2] *= -1
    theta_t = np.pi / 4.
    device = verts.device
    rot = torch.tensor([
        [np.cos(theta_t), np.sin(theta_t)],
        [-np.sin(theta_t), np.cos(theta_t)]], dtype=torch.float32, device=device)
    verts[..., [0, 2]] = verts[..., [0, 2]] @ rot.T
    n = torch.tensor([-1. / 2 ** 0.5, 0., 1. / 2 ** 0.5])
    theta_t = -np.pi / 6.
    rot = torch.tensor([
        [n[0] ** 2 * (1. - np.cos(theta_t)) + np.cos(theta_t),
         n[0] * n[1] * (1. - np.cos(theta_t)) + n[2] * np.sin(theta_t),
         n[0] * n[2] * (1. - np.cos(theta_t)) - n[1] * np.sin(theta_t)],
        [n[0] * n[1] * (1. - np.cos(theta_t)) - n[2] * np.sin(theta_t),
         n[1] ** 2 * (1. - np.cos(theta_t)) + np.cos(theta_t),
         n[1] * n[2] * (1. - np.cos(theta_t)) + n[0] * np.sin(theta_t)],
        [n[0] * n[2] * (1. - np.cos(theta_t)) + n[1] * np.sin(theta_t),
         n[1] * n[2] * (1. - np.cos(theta_t)) - n[0] * np.sin(theta_t),
         n[2] ** 2 * (1. - np.cos(theta_t)) + np.cos(theta_t)]],
        dtype=torch.float32, device=device)
    verts = verts @ rot.T
    verts = verts.flatten(start_dim=2)
    return verts


def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):
  """
  Visualize a 2d skeleton with 32 joints
  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == 64, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (32, -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 300 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
  else:
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

  ax.set_aspect('equal')
