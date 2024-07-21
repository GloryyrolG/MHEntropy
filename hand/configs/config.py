import argparse, yaml
from utils import rand_model_name
from yacs.config import CfgNode as CN

def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)

default_model_path = './model/' + rand_model_name() + '/' 
default_pretrain_model = default_model_path + 'pretrain.pth'
default_final_model = default_model_path + 'final.pth'

cfg = CN()

cfg.dataset = CN()
cfg.dataset.dataset_name = 'rhd'
cfg.dataset.image_size = [256, 256]
cfg.dataset.range_ = [[-5., -5., -5.], [5., 5., 5.]]
cfg.dataset.pe = '3d'
cfg.dataset.jointN = 21

cfg.model_dir = default_model_path
cfg.pretrain_model = default_pretrain_model
cfg.final_model = default_final_model
cfg.info_interval = 200
cfg.save_interval = 5
cfg.eval_interval = 1
cfg.eval_mscoco = False

cfg.training = CN(new_allowed=True)
cfg.training.mode = 'pretrain'
cfg.training.seed = None
cfg.training.view_correction = True
cfg.training.batch_size = 32
cfg.training.num_workers = 32
cfg.training.pth = None
cfg.training.load_mod_names = None
cfg.training.epochs = 80
cfg.training.lr = 1e-4
cfg.training.milestones = [30, 60]
cfg.training.warmups = 0
cfg.training.criterion = 'ELBOLoss'

cfg.network = CN(new_allowed=True)
# Enc.
cfg.network.enc_type = 'BasicEnc'
cfg.network.num_latent = 64
cfg.network.nums_latent = None
cfg.network.backbone = 'resnet18'
cfg.network.resnet_pretrained = True
cfg.network.conditional_p = False
cfg.network.conditional_i = False
cfg.network.feat_dim = None
cfg.network.acts = 'exp'
cfg.network.deterministic = False
# Dec.
cfg.network.iterative_refinement = False
cfg.network.decoder_type = 'mano'
cfg.network.pgm = None
cfg.network.p_nf = None
cfg.network.p_nf_dim = 3
cfg.network.tsfm_on = None
cfg.network.cond_mapping_dims = None
cfg.network.kemb = False
cfg.network.h_dims = [64, 64]
cfg.network.num_steps = 3  # NFs
cfg.network.nf_res = None
cfg.network.ddpm = False

cfg.loss = CN()
cfg.loss.kl = 0.0001


def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args = parser.parse_args()
    print(args, end='\n\n')
    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file

if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    save_dict_to_yaml(cfg, 'example.yaml')
    print(cfg)
    print(cfg_file)