# 2D-Pose
The code for cvpr-2024 pose work
### 除152以外 在其他机器上跑代码前，需要额外安装的：

1.安装sklearn包：pip install scikit-learn

2.新版本会提示warning信息，根据warning信息cd到对应目录，将对应的两个参数从None 改为True

3.从152的机器scp 代码到自己的机器上（pose_estimation和lib下的全部文件，experiments下的256x192_COCO1K_PoseCons.yaml 我新加了一些配置，另外，下图data中的文件也要scp到自己的机器上）


![image-20231021204925580](https://github.com/wangnaihao/2D-Pose/assets/82216522/b6376c8f-e6b0-4426-b24d-c7a8970185c3)


——————————————————————————————————————

3.文章目前所有新加的东西：

I. Co-train

II.reversal optimization（dst）

思想：其实这个没有完全按照dst的方法来做，根据gmm分类，伪标注中接近一半的都不正确，所以对于标签就没有选取错误标签，因为错的太多会导致模型塌陷，在基线方法的论文中有解释，dst的加入确实提升了效果

III. easy-hard mask aug（另外补的创新点）:

思想：先统计每一张img中所有热图的得分总和，用gmm拟合划分为简单和难样本，对于简单样本提升难度，用mask joint（或者用学姐的adaptive mask），对于难的参考cvpr的文章的方法，不同的是从简单的图像中裁剪关节点粘贴到难样本上，来降低难度。（貌似有效）

IV.对于WEIGHT的更新（舍弃）

不同于分类任务，pose的标签中有对于weight的加权，cvpr2023的那篇文章对weight做了重加权，我猜想是用于控制标签置信度的，来过滤掉拟合一些噪声标签，所以我后面也想试试这个重加权，目前考虑的就是暴力的把gmm的概率送进去加权。



#### How to run:

![image-20231021202347183](https://github.com/wangnaihao/2D-Pose/assets/82216522/936ad79c-68c9-4b26-a474-67ea1fc739ce)


**cotrian:**一个网络先训，令一个网络再训，这样时间会比较慢，但是经过测试，是能涨点的。

**cotrain1：**两个网络一起训，还没来得及测试，时间会比第一个少很多。

**train_2p:**  在网络中加入了warm_up的阶段，也就是可以先不用无标注数据进行训练，测试过一次，因为1k会很快过拟合，所以我warm过1个epoch，但是效果不太好，所以不建议跑这个，而且这个是单网络，有兴趣可以调试看看。

![image-20231021202741684](https://github.com/wangnaihao/2D-Pose/assets/82216522/fff51cc8-9935-448d-a4a0-c7e7be129bb7)


参数可以在配置文件的TRAIN下修改。

现在可以命令行传参了 具体添加参数为：

![image-20231021203924761](https://github.com/wangnaihao/2D-Pose/assets/82216522/e823b15f-2691-4c4e-bc14-9a9923ef437e)


另外需要安装wandb，我配置了ap可视化的代码，方便查看结果

pip install wandb

然后去官网复制密钥，命令行输入 wandb login ，贴进去密钥就可以了



--ema:是否使用ema，1为是，0为否

--reweight：没用 不用传，效果不好

--pose：loss_pose的权重,不传就是默认传进去配置文件的参数，推荐在这里传，因为wandb会记录你的参数，方便后期查看。

--worst，mix，cons：同上

--name：在wandb中记录日志名字的参数，建议写成：1k+ema+worst类似的，后面理解起来比较容易，一定要起名字！！！不然后面根本找不到



如果不传在config修改也可以，但是name一定要起，还要加上参数方便后面看，综上我觉得传参更方便一些



#####  I. 1k 5k 10k普通运行：

和以前一样，python 对应的上文中介绍的train文件就可以了。

**II**.all coco + WILD

配置文件修改：

![image-20231021204707426](https://github.com/wangnaihao/2D-Pose/assets/82216522/965b9763-0ce7-4be3-81bf-66e7248a04d1)

![image-20231021205047452](https://github.com/wangnaihao/2D-Pose/assets/82216522/3fa61514-ac4e-412f-a3bf-ef9b70ebf53b)


共计四处，另外，如果普通运行，**请一定要把TRAIN_UNSUP_SET置空**

其他没有变化，正常运行python文件就可以。
