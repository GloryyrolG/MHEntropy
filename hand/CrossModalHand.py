from argparse import Namespace
from easydict import EasyDict
import os,torch
from torch import nn
from tensorboardX import SummaryWriter
from dataloader.ho3d_dataloader import Generate_ho3d_uv
from utils import (
    init_fn, Mode, AverageMeter, get_logger,
    batch_normalize_pose3d, get_coord, crop2xyz
)
from dataloader.dataset_transforms import target_transform
from network import *
from flows import RealNVP, _reshape_inputs, combine_flow_cond
from ManoLayer import ManoLayer
from criteria import *


class CrossModalHand():
    def __init__(self, opts):
        self.opts = opts
        # Device
        self._step = 0  # training steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_name = opts.dataset.dataset_name
        self.pe = opts.dataset.pe
        self.jointN = opts.dataset.jointN

        self.batch_size = opts.training.batch_size
        self.num_workers = opts.training.num_workers
        self.view_correction = opts.training.view_correction
        self.model_path = opts.model_dir
        self.info_interval = opts.info_interval
        self.save_interval = opts.save_interval
        self.eval_interval = opts.eval_interval

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.decoder_type = opts.network.decoder_type
        self.num_latent = opts.network.num_latent # 64 as default
        self.resnet_pretrained = opts.network.resnet_pretrained
        self.deterministic = opts.network.deterministic
        self.conditional_p = opts.network.conditional_p
        self.conditional_i = opts.network.conditional_i

        # Optim args.
        self.epochs = opts.training.epochs
        self.lr = opts.training.lr
        self.milestones = opts.training.milestones

        # define encoder
        network_cfg = opts.network
        common_cfg = EasyDict(
            n_latent=network_cfg.nums_latent if network_cfg.nums_latent else self.num_latent,
            backbone=network_cfg.backbone, pretrained=self.resnet_pretrained,
            conditional_p=self.conditional_p, K=self.jointN, D=int(self.pe[0]),
            feat_dim=network_cfg.feat_dim, sigma_act=network_cfg.acts,
            deterministic=self.deterministic, input=network_cfg.input
        )
        special_cfg = EasyDict()
        if network_cfg.enc_type in ['MHEnt']:
            # Fixed cfg.
            special_cfg.update(EasyDict(
                q_z_giv_i_model=network_cfg.regressor,
                q_z_giv_i_cfg=EasyDict(
                    dim=45, tsfm_on=network_cfg.num_latent, kemb=False, jointN=self.jointN,
                    h_dims=network_cfg.h_dims, num_steps=network_cfg.num_steps),
                ds=opts.dataset.dataset_name,
                image_size=opts.dataset.image_size,
                mano_cfg=EasyDict(
                    # PCA.
                    flat_hand_mean=False, ncomps=45, use_pca=True),  # use the PCA prior
                    # flat_hand_mean=False, ncomps=45, use_pca=True),  # FreiHand MANO. The ori use_pca=False
                    # AA.
                    # flat_hand_mean=True, ncomps=45, use_pca=False),  # needs to use together w/ other priors
                    # flat_hand_mean=False, ncomps=45, use_pca=False),  # very old ver
                prior_cfg=EasyDict(p_theta45_pth=network_cfg.rot_prior, th45_ref_alpha=network_cfg.w_reg_th),
                    # p_theta45_pth='tmp/model_finished.pth'),
                data_prior_cfg=EasyDict(
                    b_2d=network_cfg.b_2d, w_prior_2d=network_cfg.w_prior_2d),
                loss_cfg=EasyDict(
                    entropy=network_cfg.entropy, mode=network_cfg.mode, w_reg_ds=network_cfg.w_reg_ds),
                kld_w=1, kld_w_annealing=[1, 20 * 1200], T=1.))
        self.encoderRGB = eval(f'{network_cfg.enc_type}')(special_cfg, **common_cfg).to(self.device)
        # self.enc_feat_register = SaveFeatures(self.encoderRGB.res.fc)

        self.use_uvd = False

        # NFs.
        self.pgm = opts.network.pgm
        self.tsfm_on = opts.network.tsfm_on
        if opts.network.p_nf == 'realnvp':
            self.p_nf = RealNVP(
                dim=opts.network.p_nf_dim, tsfm_on=opts.network.tsfm_on,
                kemb=opts.network.kemb, jointN=self.jointN,
                h_dims=opts.network.h_dims, num_steps=opts.network.num_steps,
                cond_mapping_dims=opts.network.cond_mapping_dims).to(self.device)
        elif opts.network.p_nf == 'glow':
            self.p_nf = ConditionalGlow(
                opts.network.p_nf_dim, opts.network.h_dims[0], 4, 2, dropout_probability=0,
                context_features=opts.network.tsfm_on,
                batch_norm_within_layers=True).to(self.device)
        self.nf_res = opts.network.nf_res

        # define decoder
        if self.decoder_type == 'mano':
            if self.conditional_i:
                raise NotImplementedError
            
            self.decoderPose = ManoLayer(n_latent=self.num_latent, skeidx='RHD').to(self.device)
        elif self.decoder_type == 'id':
            self.decoderPose = nn.Identity()  # none

        self.weight_kl = opts.loss.kl
        if self.deterministic and self.weight_kl != 0.:
            raise ValueError

        self.log = get_logger(self.model_path + '/info_{}.log'.format(opts.training.mode))
        self.log.info(opts)

        # Resume the model. After the log created.
        if opts.training.pth is not None:
            self.load_model(
                opts.training.pth, mod_names=opts.training.load_mod_names)
        self.warmups = opts.training.warmups

        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        with open(os.path.join(self.model_path, 'models.txt'), 'w') as fp:
            fp.write(str(self.encoderRGB) + '\n')
            fp.write(str(self.decoderPose) + '\n')
            fp.write(str(count_params(self.encoderRGB)) + '\n')
            fp.write(str(count_params(self.decoderPose)) + '\n')
            for mod in ['p_nf']:
                if mod not in self.__dict__:
                    continue
                mod = eval(f'self.{mod}')
                fp.write(str(mod) + '\n')
                fp.write(str(count_params(mod)) + '\n')

        self.tb_writer = SummaryWriter(self.model_path)

        criterion_kwargs = {'pe': self.pe}
        if self.dataset_name in ['rhd', 'freihand']:
            criterion_kwargs.update({
                'thr': 15. / 40.,
                'ds_type': 'hand',
            })
        else:
            criterion_kwargs.update({
                'thr': 0.5,
                'ds_type': 'human',
                'hm_shape': [64, 64, 64],
            })

        if opts.training.criterion == 'MHEntLoss':
            criterion_kwargs['thr'] = -1.
            self.criterion = MHEntLoss(
                loss_weights={'th_norm': 0., 'bt_norm': 0.})

    def make_ds_dl(
            self, dataset_name='rhd', mode='evaluation', shuffle=False,
            batch_size=32):
        # Hand datasets.
        if dataset_name == 'ho3d':
            dataset = Generate_ho3d_uv(mode=mode)
        elif dataset_name == 'ycb':
            pass

        else:
            raise NotImplementedError

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=init_fn)
        return dataset, dataloader

    def train_baseline(self, shuffle=True):
        _, dataloader = self.make_ds_dl(
            dataset_name=self.dataset_name, mode='training', shuffle=shuffle,
            batch_size=self.batch_size)
        eval_batch_size = 64
        eval_dataset, eval_dataloader = self.make_ds_dl(
            dataset_name=self.dataset_name, mode='evaluation', batch_size=eval_batch_size,
            shuffle=False)

        # Needs to modify the optimizer manually!
        params = []
        lr = self.lr 
        params.append({'params': self.encoderRGB.parameters(), 'lr': lr})
        params.append({'params': self.decoderPose.parameters(), 'lr': lr})
        for mod_name in ['p_nf']:
            if mod_name not in self.__dict__:
                continue
            mod = eval(f'self.{mod_name}')
            params.append({'params': mod.parameters(), 'lr': lr})
        self.optimizer = torch.optim.Adam(params)
        milestones = self.milestones
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        # Basic test.
        self.eval(dataloader=eval_dataloader, dataset=eval_dataset)
        # self.eval(split='training')

        for epoch in range(self.epochs):
            self.R2P(dataloader, Mode.Train, epoch)
            if (epoch + 1) % self.eval_interval == 0:
                self.eval(epoch=epoch, dataloader=eval_dataloader, dataset=eval_dataset)
            self.scheduler.step()

            name = 'baseline_{}'.format(self.decoder_type)
            if epoch % self.save_interval == 0:
                self.save_model(name, epoch)
        if self.epochs:
            self.save_model('baseline_{}'.format('final'))

    def model_forward(self, image, target, mode=Mode.Train) -> dict:
        output = {}

        if self.pe == '3d':
            pose_gt = target['pose3d']
        elif self.pe == '2d':
            pose_gt = target['crop_uv']

        if not getattr(self.encoderRGB, 'integrated', False):
            enc_kwargs, dec_kwargs = {}, {}
            if self.conditional_p:
                enc_kwargs['p'] = target['pose3d']
            z_rgb, mn_rgb, sd_rgb = self.encoderRGB(image, **enc_kwargs)  # encode rgb
        
            if self.decoder_type == 'mano':
                prediction = self.decoderPose(z_rgb)
                pose_rgb = prediction['mano_joints']

                # re-normlized for rhd idx root_idx=mmcp norm_idx=mpip
                # pose_rgb = batch_normalize_pose3d(pose_rgb, root_idx = 3, norm_idx = 12)
                pose_rgb = batch_normalize_pose3d(pose_rgb, root_idx = 12, norm_idx = 11)
                pose_rgb = pose_rgb.reshape(-1,63)
            
            elif self.decoder_type == 'mlp':
                pose_rgb = self.decoderPose(z_rgb, **dec_kwargs)
            
            elif self.decoder_type == 'id':
                    pose_rgb = z_rgb
                    # Here, we a bit abuse logvar for feats as well. Thus, they
                    # may have diff shapes.
                    mu, logvar = mn_rgb, 2. * sd_rgb.log()

                    if self.lifting:
                        logvar = self.encoderRGB.feat_extractor(target['crop_uv'])

            #TODO: integrate into each module for loss calculation.
            # Sampling (for cVAEs).
            if self.conditional_i and not self.deterministic:
                z_rand = torch.randn_like(z_rgb) * 0.7  # temp=0.7
                output['pose_rgb_sample'] = self.decoderPose(z_rand, **dec_kwargs)

            if 'p_nf' in self.__dict__:
                if self.pe == '3d':
                    # vis = target['target_uvd_weight']
                    vis = None
                else:
                    vis = target['target_uv_weight']
                
                # Process the obtained output to get the input.
                if self.pgm == 'inv_prob':
                    if self.p_nf.dim > 3:
                        raise NotImplementedError
                    logvar = combine_flow_cond(z=z_rgb, f=self.encoderRGB._feat)

                # Smooth the rel pose, esp for the root.
                tld_pose_gt = pose_gt + torch.randn_like(pose_gt) * 1e-4
                if isinstance(self.p_nf, RealNVP):
                    output['log_p'] = output['log_phi'] = self.p_nf.log_prob(
                        tld_pose_gt, mu=mu, logvar=logvar, weights=vis)
                elif isinstance(self.p_nf, ConditionalGlow):
                    output['log_p'] = self.p_nf.log_prob(
                        tld_pose_gt, context=logvar)[0]  # logvar: feat

                large_loss_ids = torch.nonzero(output['log_p'] < -5000, as_tuple=True)[0]
                if large_loss_ids.shape[0]:
                    if torch.any(target['action'][large_loss_ids] != 5):
                        import warnings; warnings.warn('')

                # Sample multiple times to get the mean.
                # P(x') \propto Q(x')G(x'), not easy to sample.
                K1 = 10
                if isinstance(self.p_nf, RealNVP):
                    samples = [self.p_nf.sample(
                        pose_gt.shape[0] * pose_gt.shape[1] // self.p_nf.dim,
                        mu=mu, logvar=logvar, temp=0.8) for _ in range(K1)]
                    samples = torch.cat(samples).reshape(K1, *samples[0].shape).detach()
                elif isinstance(self.p_nf, ConditionalGlow):
                    noise_temp = torch.randn(
                        image.shape[0], K1, *self.p_nf._distribution._shape,
                        dtype=torch.float32, device='cuda') * 0.8
                    samples = self.p_nf.sample_and_log_prob(
                        K1,
                        # noise=None,
                        noise=noise_temp, context=logvar)[0].permute((1, 0, 2)).detach()
                output['pose_rgb_sample'] = samples.mean(0)
                output['sigma_i'] = samples.std(0).mean()  # only consider the diag. Mean over imgs, joints, and dims
                if getattr(self.p_nf, 'tsfm_on', None) == 'x':
                    output['pose_rgb_mu'] = mu
                    output['pred_jts'] = output['pose_rgb_mu']
                else:
                    output['pred_jts'] = output['pose_rgb_sample']
                    if self.dataset_name == 'human3.6m':  # rel xyz -> rel uvd
                        # For RLE's provided evaluation. Do a quick selection.
                        jpe = samples - target['pose3d']
                        jpe = jpe.reshape(*jpe.shape[: -1], -1, 3)
                        bh_ids = jpe.norm(p=2, dim=-1).mean(-1).min(0)[1]  # shape: (B,)
                        rel_xyz = torch.gather(
                            samples, 0,
                            bh_ids[None, :, None].repeat(1, 1, samples.shape[-1])
                            )[0]
                        rel_xyz = rel_xyz.reshape(*rel_xyz.shape[: -1], -1, 3)
                        abs_xyz = (rel_xyz
                                   + target['pose3d_root'][:, None, :]) * 1000
                        uvd = rel_xyz.clone()
                        uvd[..., : 2] = (
                            target['st_cam'][:, None, : 2]
                            / (abs_xyz[..., [2]] + 1e-16) * abs_xyz[..., : 2]
                            + target['st_cam'][:, None, -2:])
                        uvd = uvd / 2  # [-1, 1) -> [-0.5, 0.5); / 1000 -> / 2000
                        output['pred_jts'] = uvd.flatten(-2)
                
                if self.pe == '3d':
                    output['xyz'] = samples
                elif self.pe == '2d':
                    output['uv'] = (samples + 1) / 2 * 256

                if self.nf_res == 'rle':
                    if self.p_nf.tsfm_on == 'x':
                        bar_mu = (pose_gt - mu) / torch.exp(0.5 * logvar)
                    else:
                        raise NotImplementedError
                
                    # Use a Gaussian as Q.
                    bar_mu_, vis_ = _reshape_inputs([self.p_nf.dim], bar_mu, vis)
                    output['log_q'] = (self.p_nf.prior.log_prob(
                        bar_mu_) * vis_[:, 0]).view(mu.shape[0], -1).sum(1)
        
            ###
        else:
            if self.opts.network.input == 'pose':
                input = target['crop_uv'] * (target['vis'] == 1).repeat_interleave(2, dim=1)
            else:
                input = image
            mods = ['xyz', 'uv']
            output = self.encoderRGB.get_loss(
                input, target, div_type=0, mods=mods)
            with torch.no_grad():
                N = self.opts.training.test_samples
                samples = self.encoderRGB.sample(
                    input, N=[N, N], temp=0.8, mods=set(mods + ['xyz', 'verts']), y=target)  # to compute the metrics
                output.update(samples)
                # print("> Debuggin. temp=1.")

        ### Criteria.
        if isinstance(self.criterion, MHEntLoss):
            if 'log_p' not in output:
                output['log_p'] = -(pose_rgb - pose_gt).abs().sum(1)  # l1 applied

                output['xyz'] = pose_rgb[None, ...].detach()

        ###
        return output

    def eval(self, name=None, dataloader=None, dataset=None, epoch=0, split='evaluation'):
        if name is not None: self.load_model(name)

        if dataloader is None:
            eval_batch_size = 64
            dataset, dataloader = self.make_ds_dl(
                dataset_name=self.dataset_name, mode=split, batch_size=eval_batch_size,
                shuffle=False)

        with torch.no_grad():
            metrics = {}
            epoch_outputs = self.R2P(dataloader, Mode.Eval, epoch)
            # For uvd.
            for k, v in metrics.items():
                self.log.info(f"> {k}: {v:.4f}")
                self.tb_writer.add_scalar(
                    f'metric_eval/{k}', v, global_step=self._step)

    def set_model_modes(self, train=False):
        for mod_name in [
                'encoderRGB', 'decoderPose', 'p_nf']:
            if mod_name not in self.__dict__:
                continue
            if train:
                eval(f'self.{mod_name}.train()')
            else:
                eval(f'self.{mod_name}.eval()')

    def R2P(self, dataloader, mode: Mode, epoch: int):
        loss_total = AverageMeter()
        eval_meters = {}
        for sup_s in ['3d', '2d']:
            vis_metrics = []
            for metric_s in ['vis', 'invis']:
                for submetric_s in ['', 'mean', 'std']:
                    conn = '_' if submetric_s else ''
                    vis_metrics.append(f'{metric_s}{conn}{submetric_s}')
            for metric_s in (['', 'mu'] + vis_metrics):
                conn = '_' if metric_s else ''
                eval_meter_s = f'eval_{sup_s}_rgb{conn}{metric_s}'

                eval_meters[eval_meter_s] = AverageMeter()
            eval_meters[f'eval_{sup_s}_rgb_sample'] = eval_meters[f'eval_{sup_s}_rgb']  #TODO: currently a bit abused for 2d as well
        eval_meters['eval_mesh_rgb_sample'] = eval_meters['eval_mesh_rgb'] = AverageMeter()

        self.set_model_modes(train=mode == Mode.Train)

        # To store some preds.
        epoch_outputs = {
            'kpt_pred': {}
        }
        if mode == Mode.Eval and self.dataset_name == 'human3.6m':
            _cfg = Namespace(**{
                'TEST': {
                    'HEATMAP2COORD': 'coord',
                },
                'DATA_PRESET': Namespace(**{
                    'IMAGE_SIZE': [256, 256],
                    'HEATMAP_SIZE': [64, 64],
                })
            })
            heatmap_to_coord = get_coord(
                _cfg, _cfg.DATA_PRESET.HEATMAP_SIZE, output_3d=True)

        for idx, data in enumerate(dataloader):

            image, target = target_transform(data, self.dataset_name)

            #TODO: still use pose3d in sanity-check, conditional_p,

            image, target = image.to(self.device),self.send_to_device(target)

            mode_s_t = {Mode.Train: "training", Mode.Eval: "validation"}[mode]
            start_func = f'{mode_s_t}_step_start'
            if getattr(self.encoderRGB, start_func, None):
                eval(f'self.encoderRGB.{start_func}({self._step})')
            output = self.model_forward(image, target, mode=mode)

            total_loss, losses, metrics = self.criterion(output, target)

            if mode == Mode.Train:
                self.optimizer.zero_grad()

                try:
                    total_loss.backward()
                except:
                    1
                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.encoderRGB.parameters(), 1.
                    )
                    if getattr(self, 'p_nf', None) is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.p_nf.parameters(), 1)
                except Exception:
                    pass
                self.optimizer.step()

            ### For evaluation.
            loss_total.update(total_loss.item())
            for sup_s in ['3d', '2d', 'mesh']:
                for _ in [
                        'mu', 'sample',
                        'vis', 'invis',
                        'vis_mean', 'invis_mean',
                        'vis_std', 'invis_std']:
                    if f'eucLoss_{sup_s}_rgb_{_}' not in metrics:
                        continue
                    avgmeter = eval_meters[f'eval_{sup_s}_rgb_{_}']
                    avgmeter.update(metrics[f'eucLoss_{sup_s}_rgb_{_}'].mean().item())

            # Store results.
            if mode == Mode.Eval:
                for i in range(image.shape[0]):
                    if self.dataset_name == 'human3.6m':
                        img_ids, bboxes = data[2:]
                        bbox = bboxes[i].tolist()
                        # Coord tsfm.
                        pose_coords, _pose_scores = heatmap_to_coord(
                            self._output_tsfm2rle(output), bbox, idx=i)
                        assert pose_coords.shape[0] == 1

                        epoch_outputs['kpt_pred'][int(img_ids[i])] = {
                            'uvd': pose_coords[0]
                        }
            
            # Logs and display.
            str = ('Epoch:{}| Step:{}| Avg_Loss:{:.4f}| eval_3d_rgb:{:.4f}|'
                    .format(epoch, idx, loss_total.avg, eval_meters['eval_3d_rgb'].avg*1000))
            if 'eucLoss_rgb_mu' in metrics:
                str += f' eval_3d_rgb_mu:{eval_meters["eval_3d_rgb_mu"].avg * 1000.:.4f}|'
                self.tb_writer.add_scalar(
                    f'metric_{["train", "eval"][mode == Mode.Eval]}/eval_3d_rgb_mu',
                    eval_meters['eval_3d_rgb_mu'].avg * 1000., global_step=self._step)  # also use train step for eval
            for sup in ['3d', '2d']:
                for attr in [
                        'vis', 'invis',
                        'vis_mean', 'invis_mean',
                        'vis_std', 'invis_std']:
                    key = f'eval_{sup}_rgb_{attr}'
                    avg = eval_meters[key].avg
                    if sup == '3d':
                        avg = avg * 1000
                    if avg:  # in most cases, err > 0
                        str += f' {attr}_{sup}:{avg:.4f}|'
            if eval_meters['eval_mesh_rgb'].avg:
                str += f' mesh: {eval_meters["eval_mesh_rgb"].avg:.4f}|'
            
            if mode == Mode.Train and idx % self.info_interval == 0:
                if self.conditional_i and not self.deterministic:
                    str += f' eval_3d_rgb_sample_it:{metrics["eucLoss_rgb_sample"].item() * 1000.:.4f}|'

                for _ in ['', '_mu', '_sample']:
                    if f'pck@50{_}' not in metrics:
                        continue
                    v = metrics[f'pck@50{_}']
                    str += f' pck@50{_}:{v.item():.4f}|'
                    self.tb_writer.add_scalar(
                        f'metric_it/pckat50{_}', v, global_step=self._step)

                self.log.info(str)

                self.tb_writer.add_scalar(
                    'loss_avg/loss_total', loss_total.avg,
                    global_step=self._step)
                
                self.tb_writer.add_scalar(
                    'metric_train/eval_3d_rgb', eval_meters['eval_3d_rgb'].avg * 1000.,
                    global_step=self._step)

                if self.decoder_type == 'mano':
                    for param_name in ['beta', 'theta']:
                        param = eval(f'prediction["{param_name}"]')
                        self.tb_writer.add_scalar(
                            f'param/{param_name}_norm',
                            torch.norm(param, dim=1).mean(),
                            global_step=self._step
                        )

            if mode == Mode.Train:
                self._step += 1

        self.log.info(str)

        if mode == Mode.Train:
            self.tb_writer.add_scalar(
                'loss_avg/loss_total', loss_total.avg, global_step=self._step)
                
        self.tb_writer.add_scalar(
            f'metric_{["train", "eval"][mode == Mode.Eval]}/eval_3d_rgb',
            eval_meters['eval_3d_rgb'].avg * 1000., global_step=self._step)

        return epoch_outputs

    @staticmethod
    def kl_loss(z_mean, z_stddev, goalStd=1.0):
        latent_loss = 0.5 * torch.sum(z_mean ** 2 + z_stddev ** 2 - torch.log(z_stddev ** 2) - goalStd, 1)
        return latent_loss

    def save_model(self, name, epoch=None):
        if epoch is None:
            pth_path = os.path.join(self.model_path, '{}.pth'.format(name))
        else:
            pth_path = os.path.join(self.model_path, '{}_{}.pth'.format(name,epoch))
        statistics = {
            'decoderPose':  self.decoderPose.state_dict(),
            'encoderRGB':  self.encoderRGB.state_dict()}
        for mod_name in ['p_nf']:
            if mod_name not in self.__dict__:
                continue
            mod = eval(f'self.{mod_name}')
            statistics[mod_name] = mod.state_dict()
        torch.save(statistics, pth_path)
        self.log.info('save model in {}'.format(pth_path))

    def load_model(self, pth_path, mod_names=None):
        check_point = torch.load(pth_path, map_location=self.device)

        if mod_names is None:
            mod_names = ['encoderRGB', 'decoderPose', 'p_nf']
        for mod_name in mod_names:
            if mod_name not in self.__dict__:
                continue
            mod = eval(f'self.{mod_name}')
            try:
                mod.load_state_dict(check_point[mod_name])
            except RuntimeError as e:
                print(e)
        self.log.info('load model from {}'.format(pth_path))

    def send_to_device(self, input: dict, device=None):
        if device is None:
            device = self.device
        for (k, v) in input.items():
            if isinstance(v, torch.Tensor):
                input[k] = v.to(device)
        return input
