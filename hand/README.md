# MHEntropy on HO3D

[<img src="../assets/teaser.png" width="300"/>]()

## Installation

The code was tested with `pytorch=1.8.0, py3.8, cuda11.1`. The env required to run the code can be installed by:

```bash
conda env create -f environment.yml  # run the cmd under ./hand/
```

## HO3D V3 Dataset Preparation

**NOTE:** It seems that several images in the HO3D dataset are incorrectly annotated.

Please download the HO3D V3 dataset from the [official website](https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX?), and arrange it in `./datasets/` according to [the snippet](https://github.com/GloryyrolG/MHEntropy/blob/master/hand/dataloader/ho3d_dataloader.py?plain=1#L21):

```python
data_root = './datasets/HO3D_v3/HO3D_v3/'
ycb_root =  './datasets/HO3D_v3/models/'
gt_root = './datasets/HO3D_v3/HO3D/data/'
seg_root = './datasets/HO3D_v3/'
```

**Visibility** is annotated in the `data.Dataset`.

## MHEntropy

[<img src="../assets/framework.png" width="800"/>]()![]()

### Entropy Loss

```python
h_q_z_giv_i = -self._reverse_log_q(z, feat.repeat(N, 1), **kwargs)
```

## Evaluation

A pre-trained model checkpoint is available [here](https://drive.google.com/drive/folders/1T-oE9PBbgnucjJeocEg1RDgcDtXbDH-M?usp=sharing). Save it under `./model/`. Run evaluation by:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-12 python run.py --cfg configs/ho3d.yaml
```

to reproduce the results:

| MPJPE | AH | 2D Vis PJD | 3D Occ PJD | RD |
| ----- | ----- | ---- | ----- | ---- |
| 20.55 | 16.95 | 3.30 | 11.93 | 0.28 |

`AH`: All Hypos, 2D EPE (pix);

`PJD`: Per Joint Diversity; 2D Vis is the most certain while 3D Occ is the most uncertain including common occlusion & depth ambiguity;

`RD`: Relative Diversity.

### Qualitative Visualization

You could leverage any [3D viewer VSCode extensions](https://github.com/stef-levesque/vscode-3dviewer) and [MeshLab](https://www.meshlab.net/) GUI to interact with 3D models.

## Acknowledgements

Thank [manopth](https://github.com/hassony2/manopth) for their awesome repos.

It is recommended to download MANO from the [official website](https://mano.is.tue.mpg.de/) though we provide it.
