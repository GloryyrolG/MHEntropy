import torch
from configs.config import parse_args
from CrossModalHand import CrossModalHand
from utils import setSeed

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    torch.cuda.empty_cache()
    cfg, cfg_file = parse_args()
    setSeed(seed=cfg.training.seed)

    crossHand = CrossModalHand(cfg)
    # cfg.network.decoder_type = 'mano'
    # cfg.training.mode = 'baseline_VAE'
    if cfg.training.mode == 'baseline_VAE':
        crossHand.train_baseline()
    elif cfg.training.mode == 'eval':
        crossHand.eval(name=cfg.training.pth)
