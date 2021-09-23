# 人流量统计/人体检测

## 1. 项目说明

本案例面向人流量统计/人体检测等场景，提供基于PaddleDetection的解决方案，希望通过梳理优化模型精度和性能的思路帮助用户更高效的解决实际问题。

应用场景：静态场景下的人员计数和动态场景下的人流量统计

![demo](./demo.png)

# 业务难点需要后面在模型选择的时候提一下

业务难点：

* 遮挡重识别问题。场景中行人可能比较密集，人与人之间存在遮挡问题。这可能会导致误检、漏检问题。同时，对遮挡后重新出现的行人进行准确的重识别也是一个比较复杂的问题。容易出现ID切换问题。

* 行人检测的实时性。在实际应用中，往往对行人检测的处理速度有一定要求。

  

## 2. 数据准备

### 训练数据集

请参照 [数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/docs/tutorials/PrepareMOTDataSet_cn.md) 去下载并准备好所有的数据集，包括 Caltech Pedestrian, CityPersons, CHUK-SYSU, PRW, ETHZ, MOT17和MOT16。训练时，我们采用前六个数据集，共 53694 张已标注好的数据集用于训练。MOT16作为评测数据集。所有的行人都有检测框标签，部分有ID标签。如果您想使用这些数据集，请遵循他们的License。对数据集的详细介绍参见：[数据集介绍](dataset.md)

### 数据格式

上述数据集都遵循以下结构：

```
Caltech
   |——————images
   |        └——————00001.jpg
   |        |—————— ...
   |        └——————0000N.jpg
   └——————labels_with_ids
            └——————00001.txt
            |—————— ...
            └——————0000N.txt
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train
```

所有数据集的标注是以统一数据格式提供的。各个数据集中每张图片都有相应的标注文本。给定一个图像路径，可以通过将字符串`images`替换为 `labels_with_ids`并将 `.jpg`替换为`.txt`来生成标注文本路径。在标注文本中，每行都描述一个边界框，格式如下：

```
[class] [identity] [x_center] [y_center] [width] [height]
```

注意：

* `class`为`0`，目前仅支持单类别多目标跟踪。
* `identity`是从`1`到`num_identifies`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
* `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，它们的值是基于图片的宽度/高度进行标准化的，因此值为从0到1的浮点数。

### 数据集目录

首先按照以下命令下载`image_lists.zip`并解压放在`dataset/mot`目录下：

```bash
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```

然后依次下载各个数据集并解压，最终目录为：

```
dataset/mot
  |——————image_lists
            |——————caltech.10k.val  
            |——————caltech.all  
            |——————caltech.train  
            |——————caltech.val  
            |——————citypersons.train  
            |——————citypersons.val  
            |——————cuhksysu.train  
            |——————cuhksysu.val  
            |——————eth.train  
            |——————mot15.train  
            |——————mot16.train  
            |——————mot17.train  
            |——————mot20.train  
            |——————prw.train  
            |——————prw.val
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT15
  |——————MOT16
  |——————MOT17
  |——————PRW
```



### 调优数据集

在进行调优时，我们采用 Caltech Pedestrian, CityPersons, CHUK-SYSU, PRW, ETHZ和MOT17中一半的数据集，使用MOT17另一半数据集作为评测数据集。调优时和训练时使用的数据集不同，主要是因为MOT官网的测试集榜单提交流程比较复杂，这种数据集的使用方式也是学术界慢慢摸索出的做消融实验的方法。调优时使用的训练数据共 51035 张。



## 3. 模型选择

PaddleDetection对于多目标追踪算法主要提供了三种模型，DeepSORT、JDE和FairMOT。

- [DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个ReID重识别任务。DeepSORT所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID模型此处选择[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`PCB+Pyramid ResNet101`模型。
- [JDE](https://arxiv.org/abs/1909.12605)(Joint Detection and Embedding)是在一个单一的共享神经网络中同时学习目标检测任务和embedding任务，并同时输出检测结果和对应的外观embedding匹配的算法。JDE原论文是基于Anchor Base的YOLOv3检测器新增加一个ReID分支学习embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。
- [FairMOT](https://arxiv.org/abs/2004.01888)以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

这里我们选择了FairMOT算法进行人流量统计/人体检测。



## 4. 模型训练

下载PaddleDetection

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

运行如下代码开始训练模型：

使用两个GPU开启训练

```bash
cd PaddleDetection/
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```



## 5. 模型优化(进阶)

### 5.1 精度优化

本小节侧重展示在模型优化过程中，提升模型精度的思路。在这些思路中，有些会对精度有所提升，有些没有。在其他人流量统计/人体检测场景中，可以根据实际情况尝试如下策略，不同的场景下可能会有不同的效果。

#### (1) 基线模型选择

本案例采用FairMOT模型作为基线模型，其骨干网络选择是DLA34。基线模型共有三种：

1）训练基于NVIDIA Tesla V100 32G 2GPU，batch size = 6，使用Adam优化器，模型使用CrowdHuman数据集进行预训练；

2）训练基于NVIDIA Tesla V100 32G 4GPU，batch size = 8，使用Momentum优化器，模型使用CrowdHuman数据集进行预训练；

3）训练基于NVIDIA Tesla V100 32G 4GPU，batch size = 8，使用M

并使用基于CrowdHuman数据集进行预训练。

本案例采用FairMOT模型作为基线模型，其骨干网络选择是DLA34，。模型

本案例采用FairMOT模型作为基线模型，其骨干网络选择是DLA34，并使用基于CrowdHuman数据集进行预训练。模型优化时使用的数据集，参见：`调优数据集`。（基线模型训练基于NVIDIA Tesla V100 32GB 2GPU）

| 模型             | MOTA | 推理速度 |
| ---------------- | ---- | -------- |
| Baseline (DLA34) | 70.9 |          |



#### (2) 数据增广



#### (3) 可变形卷积加入

实验结果

| 模型                                                         | MOTA | 推理速度（开启TensorRT） |
| ------------------------------------------------------------ | ---- | ------------------------ |
| baseline (dla34 2gpu bs6 adam lr=0.0001)                     | 70.9 |                          |
| baseline (dla34 4gpu bs8 momentum)                           | 67.5 |                          |
| baseline (dla34 4gpu bs8 momentum + no_pretrain)             | 64.3 |                          |
| dla34 4gpu bs8 momentum + dcn                                | 67.2 |                          |
| dla34 4gpu bs8 momentum + syncbn + ema                       | 67.4 |                          |
| dla34 4gpu bs8 momentum + cutmix                             | 67.7 |                          |
| dla34 4gpu bs8 momentum + attention                          | 67.6 |                          |
| dla34 4gpu bs6 adam lr=0.0002                                | 71.1 |                          |
| dla34 4gpu bs6 adam lr=0.0002 + syncbn + ema + attention     | 71.6 |                          |
| dla34 4gpu bs6 adam lr=0.0002 + syncbn + ema + sann          | 71.1 |                          |
| dla34 4gpu bs6 adam lr=0.0002 + syncbn + ema + attention + cutmix | 71.3 |                          |
| dla46c 4gpu bs8 momentum + no_pretrain                       | 61.2 |                          |
| dla60 4gpu bs8 momentum + no_pretrain                        | 58.8 |                          |
| dla102 4gpu bs8 momentum + no_pretrain                       | 54.8 |                          |





### 5.2 性能优化

## 6. 模型预测

## 7. 模型导出

## 8. 模型上线选择



## 引用



