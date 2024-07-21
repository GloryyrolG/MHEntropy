# The Official PyTorch Implementation of "MHEntropy: Multiple Hypotheses Meet Entropy for Pose and Shape Recovery" [(ICCV 2023 Paper)]()

[Rongyu Chen](https://gloryyrolg.github.io/), [Linlin Yang*](https://mu4yang.com/), and [Angela Yao](https://www.comp.nus.edu.sg/~ayao/)<br>
[National University of Singapore, School of Computing, Computer Vision & Machine Learning (CVML) Group](https://cvml.comp.nus.edu.sg/)<br>
ICCV 2023<br>
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_MHEntropy_Entropy_Meets_Multiple_Hypotheses_for_Pose_and_Shape_Recovery_ICCV_2023_paper.pdf) | [website](https://gloryyrolg.github.io/MHEntropy/)

Thanks for your interests.

[<img src="./assets/framework.png" width="800"/>]()

## Hands

[<img src="./assets/teaser.png" width="300"/>]()

Please find the hand experiments [here](https://github.com/GloryyrolG/MHEntropy/blob/master/hand/README.md).

## Humans

[<img src="./assets/humans.png" width="500"/>]()

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
@inproceedings{chen2023MHEntropy,
  title={{MHEntropy}: Multiple Hypotheses Meet Entropy for Pose and Shape Recovery},
  author={Chen, Rongyu and Yang, Linlin and Yao, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
