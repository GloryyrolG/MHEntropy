# The Official PyTorch Implementation of "MHEntropy: Multiple Hypotheses Meet Entropy for Pose and Shape Recovery" (ICCV 2023 Paper)

<p align="center">
  <a href="https://gloryyrolg.github.io/"><strong>Rongyu Chen</strong></a>
  ·
  <a href="https://mu4yang.com/"><strong>Linlin Yang*</strong></a>
  ·
  <a href="https://www.comp.nus.edu.sg/~ayao/"><strong>Angela Yao</strong></a><br>
  <a href="https://cvml.comp.nus.edu.sg/"><strong>National University of Singapore, School of Computing, Computer Vision & Machine Learning (CVML) Group</strong></a><br>
  ICCV 2023<br>
  <a href='https://gloryyrolg.github.io/MHEntropy/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_MHEntropy_Entropy_Meets_Multiple_Hypotheses_for_Pose_and_Shape_Recovery_ICCV_2023_paper.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href="https://www.youtube.com/watch?v=0riX3iJeVyM"><img src="https://badges.aleen42.com/src/youtube.svg"></a>
</p>

Thanks for your interest.

We mainly tackle the problem of using **only visible 2D** keypoints that are easy to annotate to train the HMR model to model **ambiguity** (occlusion, depth ambiguity, etc.) and generate multiple feasible, accurate, and diverse hypos. It also answers "*how generative models help discriminative tasks*". The key idea is that using knowledge rather than data samples to define the target data distribution under a probabilistic framework, KLD will naturally derive a missing **entropy** term.

<p align="center"><img src="./assets/framework.png" width="800"></p>

## Hands

<p align="center"><img src="./assets/teaser.png" width="300"></p>

Please find the hand experiments [here](https://github.com/GloryyrolG/MHEntropy/blob/master/hand/README.md).

## Humans

<p align="center"><img src="./assets/humans.png" width="500"></p>

Our method can be adapted to a variety of backbone models. Simply use the [ProHMR code repo](https://github.com/nkolot/ProHMR/tree/master) to load our [pre-trained model weights](https://drive.google.com/file/d/19gaxHvpTB5f6ojYECSc8uXXdLtghTxGC/view?usp=sharing) to perform inference and evaluation.

[NLL (Negative Log-likelihood) Loss](https://github.com/nkolot/ProHMR/blob/3b1a9926f97ba1c77f1cb97151da2a59d2f16d11/prohmr/models/prohmr.py#L236)
```python
log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats[has_smpl_params])
loss_nll = -log_prob.mean()
```
VS. Our Entropy Loss
```python
pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)
log_prob_ent = output['log_prob'][:, 1:]
loss_ent = log_prob_ent.mean()
```

## Bibtex:
Please cite our paper, if you happen to use this codebase:

```
@inproceedings{chen2023mhentropy,
  title={{MHEntropy}: Entropy Meets Multiple Hypotheses for Pose and Shape Recovery},
  author={Chen, Rongyu and Yang, Linlin and Yao, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
