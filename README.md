# Enhancing the Quality of Pseudo Labels in 2D Human Pose Estimation via a Debiasing-Teacher Approach⋆

#### The framework of our work：

![image](https://github.com/wangnaihao/Debias-Teacher/assets/82216522/06d67cea-5242-4c89-8a59-7c296e310fb3)

#### How to run：
```
python pose_estimation/cotrain1.py --cfg experiments/mix_coco_coco/res18/256x192_COCO1K_PoseCons.yaml --gpus 1
```
#### Acknowledgements
The code is mainly based on [Simple Baseline](https://github.com/microsoft/human-pose-estimation.pytorch) and [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). Some code comes from [DarkPose](https://github.com/ilovepose/DarkPose) , [Semi-human-pose](https://github.com/xierc/Semi_Human_Pose) and [SSPCM](https://github.com/hlz0606/SSPCM). Thanks for their works.
