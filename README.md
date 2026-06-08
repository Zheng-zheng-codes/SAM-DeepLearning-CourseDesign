# SAM-DeepLearning-CourseDesign

本项目为《机器学习与深度学习》课程设计项目，主题为 **锐度感知最小化优化算法（Sharpness-Aware Minimization, SAM）的实现与分析**。

项目基于 PyTorch 框架，在 CIFAR-10 图像分类任务上实现并比较三种优化方法：**SGD、Adam 和 SAM**。通过训练损失、测试准确率、收敛速度和训练时间等指标，分析不同优化算法对模型训练效果和泛化能力的影响。

---

## 1. 项目简介

深度神经网络的训练效果不仅与网络结构有关，也与优化算法密切相关。传统优化算法如 SGD 和 Adam 主要关注降低训练集上的损失函数值，但模型在训练集上表现良好并不一定意味着具有良好的泛化能力。

SAM 优化算法的核心思想是：不仅要寻找损失较低的参数点，还希望该参数点附近的损失也较低。换言之，SAM 倾向于寻找更加“平坦”的极小值区域，从而提升模型在测试集上的泛化能力。

本项目主要完成以下工作：

1. 搭建图像分类训练框架；
2. 实现 SimpleCNN 和 ResNet18 网络模型；
3. 实现 SGD、Adam 和 SAM 优化器训练流程；
4. 在 CIFAR-10 数据集上进行对比实验；
5. 分析不同优化算法的训练效果和泛化性能；
6. 探究 SAM 中扰动半径参数 `rho` 对实验结果的影响。

---

## 2. 项目目录结构

```text
DL-SAM-Optimizer/
├── README.md
├── requirements.txt
├── train.py
├── models/
│   ├── simple_cnn.py
│   └── resnet.py
├── optim/
│   └── sam.py
├── utils/
│   ├── dataset.py
│   ├── train_utils.py
│   └── plot.py
├── results/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── report/
    └── course_design_report.pdf
```

各目录说明如下：

| 路径                 | 说明                               |
| ------------------ | -------------------------------- |
| `train.py`         | 主训练程序，用于启动不同模型和优化器实验             |
| `models/`          | 存放网络模型代码，包括 SimpleCNN 和 ResNet18 |
| `optim/`           | 存放 SAM 优化器实现                     |
| `utils/`           | 存放数据集加载、训练辅助函数和绘图工具              |
| `results/logs/`    | 保存训练日志                           |
| `results/figures/` | 保存损失曲线、准确率曲线等实验图像                |
| `results/tables/`  | 保存实验结果表格                         |
| `report/`          | 存放课程设计报告                         |

---

## 3. 实验环境

本项目建议使用 Python 3.8 及以上版本，主要依赖如下：

```text
torch
torchvision
numpy
matplotlib
tqdm
pandas
```

推荐环境：

| 项目          | 版本                 |
| ----------- | ------------------ |
| Python      | 3.8+               |
| PyTorch     | 2.0+               |
| torchvision | 0.15+              |
| CUDA        | 可选                 |
| 操作系统        | Windows / Linux 均可 |

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 Anaconda，也可以新建虚拟环境：

```bash
conda create -n sam-course python=3.9
conda activate sam-course
pip install -r requirements.txt
```

---

## 4. 数据集说明

本项目使用 **CIFAR-10** 数据集。

CIFAR-10 是一个常用的图像分类数据集，共包含 10 个类别，每张图像大小为 32×32。程序运行时会通过 `torchvision.datasets.CIFAR10` 自动下载数据集。

数据集默认保存路径为：

```text
./data/
```

如果数据集已经下载过，程序会自动读取本地数据，不会重复下载。

---

## 5. 运行方法

### 5.1 使用 SGD 训练模型

```bash
python train.py --dataset cifar10 --model resnet18 --optimizer sgd --epochs 20 --batch_size 128 --lr 0.1
```

### 5.2 使用 Adam 训练模型

```bash
python train.py --dataset cifar10 --model resnet18 --optimizer adam --epochs 20 --batch_size 128 --lr 0.001
```

### 5.3 使用 SAM 训练模型

```bash
python train.py --dataset cifar10 --model resnet18 --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.05
```

### 5.4 测试不同 rho 参数

```bash
python train.py --dataset cifar10 --model resnet18 --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.01

python train.py --dataset cifar10 --model resnet18 --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.05

python train.py --dataset cifar10 --model resnet18 --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.1
```

---

## 6. 实验内容

本项目计划完成以下实验：

### 6.1 不同优化器对比实验

在相同数据集、相同模型结构和相同训练轮数下，对比以下优化算法：

| 优化器  | 说明           |
| ---- | ------------ |
| SGD  | 经典随机梯度下降优化算法 |
| Adam | 自适应学习率优化算法   |
| SAM  | 锐度感知最小化优化算法  |

主要比较指标包括：

1. 训练损失；
2. 测试损失；
3. 训练准确率；
4. 测试准确率；
5. 最佳测试准确率；
6. 训练时间。

### 6.2 SAM 参数分析实验

SAM 中的重要参数为 `rho`，表示参数扰动半径。本项目将比较不同 `rho` 取值对模型性能的影响：

```text
rho = 0.01
rho = 0.05
rho = 0.10
```

通过实验分析 `rho` 过小、适中和过大时对模型训练稳定性和泛化性能的影响。

### 6.3 不同模型结构对比实验

为了进一步分析 SAM 的适用性，本项目计划在两种模型上进行实验：

| 模型        | 说明       |
| --------- | -------- |
| SimpleCNN | 简单卷积神经网络 |
| ResNet18  | 残差神经网络   |

通过比较不同模型下 SAM 的表现，分析 SAM 是否在不同网络结构中都能带来泛化能力提升。

---

## 7. 实验结果保存

训练完成后，程序会自动保存实验结果。

日志文件保存到：

```text
results/logs/
```

训练曲线保存到：

```text
results/figures/
```

实验表格保存到：

```text
results/tables/
```

示例输出包括：

```text
results/logs/resnet18_sgd_cifar10.log
results/logs/resnet18_adam_cifar10.log
results/logs/resnet18_sam_cifar10_rho0.05.log

results/figures/loss_curve.png
results/figures/accuracy_curve.png
results/figures/rho_comparison.png

results/tables/result_summary.csv
```

---

## 8. 预期实验分析

本项目希望通过实验回答以下问题：

1. SAM 是否能够提升模型在测试集上的准确率？
2. SAM 与 SGD、Adam 相比，收敛速度有什么差异？
3. SAM 的训练时间是否明显增加？
4. SAM 中 `rho` 参数如何影响模型性能？
5. SAM 是否在不同网络结构中都具有较好的效果？

预期结论是：SAM 由于在训练过程中考虑了参数邻域内的最大损失，可能会获得更好的泛化性能；但由于每次参数更新需要两次前向传播和两次反向传播，因此训练时间通常会高于普通 SGD 和 Adam。

---

## 9. 课程设计报告

课程设计报告位于：

```text
report/course_design_report.pdf
```

报告主要包括以下内容：

1. 摘要；
2. 引言；
3. 相关工作；
4. 方法介绍；
5. 实验设计；
6. 实验结果与分析；
7. 结论；
8. 参考文献。

---

## 10. 参考文献

[1] Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
Sharpness-Aware Minimization for Efficiently Improving Generalization.
International Conference on Learning Representations, 2021.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[3] Alex Krizhevsky.
Learning Multiple Layers of Features from Tiny Images.
Technical Report, 2009.

---

## 11. 注意事项

1. 第一次运行程序时需要下载 CIFAR-10 数据集，请保持网络连接正常；
2. 如果没有 GPU，也可以使用 CPU 运行，但训练速度会较慢；
3. SAM 训练过程比普通优化器更慢，这是因为每次更新需要两次梯度计算；
4. 为了保证实验结果可复现，建议在程序中固定随机种子；
5. 最终提交时应包含完整代码、README 文件、实验结果和课程设计报告。

---

## 12. 项目状态

当前项目处于课程设计开发阶段，后续将继续完善：

* [ ] 完成数据集加载模块；
* [ ] 完成 SimpleCNN 模型；
* [ ] 完成 ResNet18 模型；
* [ ] 完成 SGD 和 Adam 训练流程；
* [ ] 实现 SAM 优化器；
* [ ] 完成实验日志保存；
* [ ] 完成实验结果绘图；
* [ ] 完成课程设计报告。
