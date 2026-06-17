# SAM-CourseDesign

本项目为《机器学习与深度学习》课程设计项目，选题为 **新型优化算法的实现与分析**。项目围绕图像分类任务，重点实现锐度感知最小化优化算法 **Sharpness-Aware Minimization, SAM**，并与经典优化算法 **SGD** 和 **Adam** 进行实验比较。在完成基础对比实验后，进一步对不同优化算法进行参数调优，并尝试对 SAM 优化算法进行改进。

本项目主要分为三个阶段：

1. **基础对比阶段**：实现并比较 SGD、Adam 和 SAM 三种优化算法；
2. **参数调优阶段**：分析不同学习率、动量参数和 SAM 扰动半径 `rho` 对模型性能的影响；
3. **SAM 改进阶段**：尝试设计改进版 SAM，并与原始 SAM 进行对比分析。

---

## 1. 项目背景

深度神经网络的训练效果不仅依赖于模型结构和数据集，也与优化算法密切相关。SGD 和 Adam 是深度学习中常用的经典优化方法，其中 SGD 具有较好的泛化能力，Adam 具有较快的收敛速度。

SAM 是一种面向模型泛化能力的新型优化算法。普通优化算法通常直接最小化当前参数位置的训练损失，而 SAM 不仅关注当前参数点的损失，还关注参数邻域内的最大损失。它希望模型收敛到更加平坦的极小值区域，从而提高模型在测试集上的泛化能力。

本项目以图像分类任务为实验场景，在相同网络结构和相同数据集上对比 SGD、Adam 和 SAM 的表现，并进一步研究 SAM 参数设置和改进策略对模型性能的影响。

---

## 2. 项目目标

本课程设计的主要目标如下：

1. 实现 CIFAR-10 图像分类训练流程；
2. 实现 ResNet18 图像分类模型；
3. 实现并比较 SGD、Adam 和 SAM 三种优化算法；
4. 对不同优化算法进行参数调优；
5. 分析不同优化算法在训练损失、测试准确率、收敛速度和训练时间方面的差异；
6. 探究 SAM 中扰动半径 `rho` 对模型泛化能力的影响；
7. 尝试对 SAM 优化算法进行改进；
8. 通过实验结果给出合理、可靠的分析结论。

---

## 3. 项目目录结构

```text id="a8tkj5"
SAM-DeepLearning-CourseDesign/
├── README.md
├── requirements.txt
├── datatest.py
├── train_18.py
├── train_34.py
├── train_save_model.py
├── stage3_train_new_sam.py
├── stage4_dynamic_rho_lr_improved_sam.py
├── stage5_feedback_dynamic_rho_lr_improved_sam.py
├── models/
│   └── resnet.py
├── optim/
│   ├── improved_sam.py
│   ├── improved_sam3.py
│   └── sam.py
├── utils/
│   ├── dataset.py
│   ├── train_utils.py
├── data/
│   ├── cifar-10-batches-py/
├── results/
│   ├── logs/
│   ├── figures/
│   └── tables/
```

目录说明：

| 路径                      | 说明            |
| ----------------------- | ------------- |
| `README.md`             | 项目说明文档        |
| `requirements.txt`      | 项目依赖文件        |
| `datatest.py`           | 数据集测试与验证脚本      |
| `train_18.py`           | ResNet18 单次实验训练脚本      |
| `train_34.py`           | ResNet34 单次实验训练脚本      |
| `train_save_model.py`   | 训练并保存模型脚本      |
| `stage3_train_new_sam.py` | 动态 ImprovedSAM 对比脚本 |
| `stage4_dynamic_rho_lr_improved_sam.py` | 动态 rho+LR ImprovedSAM 脚本 |
| `stage5_feedback_dynamic_rho_lr_improved_sam.py` | gap-aware 反馈 ImprovedSAM 脚本 |
| `models/resnet.py`      | ResNet 模型定义 |
| `optim/sam.py`          | 原始 SAM 优化器实现  |
| `optim/improved_sam.py` | 改进版 SAM 优化器实现 |
| `optim/improved_sam2.py` | 第二版改进 SAM 实现 |
| `optim/improved_sam3.py` | 第三版改进 SAM 实现 |
| `utils/dataset.py`      | 数据集加载与预处理     |
| `utils/train_utils.py`  | 训练与测试函数       |
| `data/`                 | 数据集目录         |
| `results/logs/`         | 训练日志保存目录      |
| `results/figures/`      | 实验图像保存目录      |
| `results/tables/`       | 实验结果表格保存目录    |

---

## 4. 实验环境

本项目基于 PyTorch 实现，推荐使用 Python 3.8 及以上版本。

主要依赖如下：

```text id="u68bhe"
torch
torchvision
numpy
matplotlib
pandas
tqdm
```

安装依赖：

```bash id="q93m1z"
pip install -r requirements.txt
```

如果使用 Anaconda，可以创建独立环境：

```bash id="lk4kkc"
conda create -n sam-course python=3.9
conda activate sam-course
pip install -r requirements.txt
```

---

## 5. 数据集说明

本项目使用 **CIFAR-10** 数据集。

CIFAR-10 是常用的图像分类数据集，共包含 10 个类别，每张图像大小为 32×32。程序运行时会通过 `torchvision.datasets.CIFAR10` 自动下载数据集。

数据集默认保存路径为：

```text id="lul6kw"
./data/
```

---

## 6. 模型结构

本项目使用 ResNet18 作为主要实验模型。

由于 CIFAR-10 图像尺寸为 32×32，项目中会对原始 ResNet18 进行适配：

1. 将第一层卷积由 `7×7, stride=2` 改为 `3×3, stride=1`；
2. 移除原始 ResNet18 中的最大池化层；
3. 将最后的全连接层输出类别数改为 10。

这样可以使 ResNet18 更适合 CIFAR-10 图像分类任务。

---

## 7. 实验设计

本项目实验分为三个阶段。

---

### 7.1 第一阶段：SGD、Adam 与 SAM 基础对比

第一阶段主要实现并比较三种优化算法：

| 实验编号 | 数据集      | 模型       | 优化器  |
| ---- | -------- | -------- | ---- |
| Exp1 | CIFAR-10 | ResNet18 | SGD  |
| Exp2 | CIFAR-10 | ResNet18 | Adam |
| Exp3 | CIFAR-10 | ResNet18 | SAM  |

该阶段主要回答以下问题：

1. SAM 是否能够在测试集上取得更好的准确率？
2. SGD、Adam 和 SAM 的收敛速度有什么差异？
3. SAM 的训练时间是否明显高于 SGD 和 Adam？
4. SAM 是否能提升模型泛化能力？

运行示例：

```bash id="edkoln"
python train.py --optimizer sgd --epochs 20 --batch_size 128 --lr 0.1

python train.py --optimizer adam --epochs 20 --batch_size 128 --lr 0.001

python train.py --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.05
```

---

### 7.2 第二阶段：优化算法参数调优

第二阶段主要对不同优化算法进行参数调优。

对于 SGD，主要分析学习率和动量参数的影响：

| 实验编号 | 优化器 | 学习率  | 动量  |
| ---- | --- | ---- | --- |
| Exp4 | SGD | 0.01 | 0.9 |
| Exp5 | SGD | 0.1  | 0.9 |
| Exp6 | SGD | 0.1  | 0.5 |

对于 Adam，主要分析学习率的影响：

| 实验编号 | 优化器  | 学习率    |
| ---- | ---- | ------ |
| Exp7 | Adam | 0.001  |
| Exp8 | Adam | 0.0005 |
| Exp9 | Adam | 0.0001 |

对于 SAM，主要分析扰动半径 `rho` 的影响：

| 实验编号  | 优化器 | 学习率 | rho  |
| ----- | --- | --- | ---- |
| Exp10 | SAM | 0.1 | 0.01 |
| Exp11 | SAM | 0.1 | 0.05 |
| Exp12 | SAM | 0.1 | 0.10 |

该阶段主要回答以下问题：

1. 不同学习率对模型收敛速度和最终准确率有什么影响？
2. SGD 中动量参数如何影响训练稳定性？
3. SAM 中 `rho` 参数过大或过小时会产生什么影响？
4. 哪一组参数能取得较好的综合表现？

运行示例：

```bash id="y5q0ax"
python train.py --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.01

python train.py --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.05

python train.py --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.10
```

---

### 7.3 第三阶段：SAM 优化算法改进

第三阶段尝试对原始 SAM 优化算法进行改进。

原始 SAM 使用固定的扰动半径 `rho`。但是在训练的不同阶段，模型对扰动强度的需求可能不同。训练前期模型参数尚未稳定，如果扰动半径过大，可能影响收敛；训练后期模型逐渐收敛，适当增强对平坦极小值的约束，可能有助于提升泛化能力。

因此，本项目先后完成了三轮改进实验：

1. **Stage3 动态 rho ImprovedSAM**：对扰动半径做线性 warmup + cosine decay；
2. **Stage4 动态 rho+LR ImprovedSAM**：在动态 rho 的基础上增加学习率调整；
3. **Stage5 gap-aware feedback ImprovedSAM**：引入训练/验证 gap 反馈机制，同时在线调整 lr 和 rho。

实验结果表明：

- `Stage2` 原始 SAM 最佳测试准确率：`0.9419`（`lr=0.01`, `rho=0.1`）；
- `Stage3` 动态 rho ImprovedSAM 最佳测试准确率：`0.9161`；
- `Stage4` 动态 rho+LR ImprovedSAM 最佳测试准确率：`0.9377`；
- `Stage5` gap-aware feedback ImprovedSAM 最佳测试准确率：`0.9412`。

从结果来看，单纯的动态 rho 改进未能超过最优原始 SAM，而在进一步加入学习率调整与 gap-aware 反馈后，改进版 SAM 的性能明显提升，能够接近甚至逼近原始 SAM 的测试准确率。

Warmup-SAM 的基本思想是：

1. 训练前期使用较小的 `rho`；
2. 随着 epoch 增加，逐渐增大 `rho`；
3. 当达到设定最大值后，保持 `rho` 不变。

动态 `rho` 的设置方式如下：

```text id="hw3vni"
rho_t = rho_min + (rho_max - rho_min) * t / T_warmup
```

其中：

```text id="dx1eq8"
rho_min = 0.01
rho_max = 0.05
T_warmup = 训练总 epoch 数的 30%
```

当训练轮数超过 `T_warmup` 后：

```text id="drrs01"
rho_t = rho_max
```

第三阶段将比较：

| 实验编号  | 数据集      | 模型       | 优化器        |
| ----- | -------- | -------- | ---------- |
| Exp13 | CIFAR-10 | ResNet18 | SGD        |
| Exp14 | CIFAR-10 | ResNet18 | SAM        |
| Exp15 | CIFAR-10 | ResNet18 | Warmup-SAM |

该阶段主要回答以下问题：

1. 改进版 Warmup-SAM 是否优于原始 SAM？
2. 动态扰动半径是否能提升训练稳定性？
3. Warmup-SAM 是否能在准确率和训练时间之间取得更好的平衡？
4. 改进方法是否具有合理性？

运行示例：

```bash id="o8724w"
python train.py --optimizer sam --epochs 20 --batch_size 128 --lr 0.1 --rho 0.05

python train.py --optimizer warmup_sam --epochs 20 --batch_size 128 --lr 0.1 --rho_min 0.01 --rho_max 0.05
```

---

## 8. 评价指标

实验主要从以下几个方面进行评价：

| 指标            | 说明           |
| ------------- | ------------ |
| Train Loss    | 训练集损失        |
| Train Acc     | 训练集准确率       |
| Test Loss     | 测试集损失        |
| Test Acc      | 测试集准确率       |
| Best Test Acc | 训练过程中最高测试准确率 |
| Epoch Time    | 单轮训练时间       |
| Total Time    | 总训练时间        |

通过这些指标，可以综合分析不同优化算法的收敛速度、分类性能、泛化能力和训练成本。

---

## 9. 输出结果

每次实验结束后，程序会保存训练日志、实验曲线和结果表格。

日志文件保存到：

```text id="pw5twk"
results/logs/
```

图像文件保存到：

```text id="q44ms7"
results/figures/
```

结果表格保存到：

```text id="0thvdm"
results/tables/
```

### 9.1 实验结果摘要

本项目最终完成的主要结果如下：

- `Stage2` 原始 SAM 最佳测试准确率：`0.9419`（`lr=0.01`, `rho=0.1`）；
- `Stage3` 动态 rho ImprovedSAM 最佳测试准确率：`0.9161`；
- `Stage4` 动态 rho+LR ImprovedSAM 最佳测试准确率：`0.9377`；
- `Stage5` gap-aware feedback ImprovedSAM 最佳测试准确率：`0.9412`。

从结果来看，单纯的动态 rho 改进未能超过最优原始 SAM，而进一步的学习率调整和 gap-aware 反馈机制显著提升了改进 SAM 的性能，使其与原始 SAM 的测试准确率相当，并增强了训练过程的鲁棒性。

示例输出：

```text id="bl1skk"
results/logs/resnet18_cifar10_sgd.csv
results/logs/resnet18_cifar10_adam.csv
results/logs/resnet18_cifar10_sam_rho0.05.csv
results/logs/resnet18_cifar10_warmup_sam.csv

results/figures/loss_comparison.png
results/figures/accuracy_comparison.png
results/figures/rho_comparison.png
results/figures/improved_sam_comparison.png

results/tables/baseline_results.csv
results/tables/rho_results.csv
results/tables/improved_sam_results.csv
```

---

## 10. 结果分析计划

课程设计报告中将重点分析以下内容：

1. **SGD、Adam 和 SAM 的性能对比**
   比较三种优化器在测试准确率、训练损失和收敛速度方面的差异。

2. **SAM 的泛化能力分析**
   分析 SAM 是否能通过寻找更平坦的极小值区域提升测试集表现。

3. **参数调优分析**
   分析学习率、动量参数和 `rho` 参数对模型训练效果的影响。

4. **SAM 改进方法分析**
   比较 Warmup-SAM 与原始 SAM 的实验结果，分析动态扰动半径是否有效。

5. **训练成本分析**
   分析 SAM 和 Warmup-SAM 由于两次前向传播和两次反向传播带来的额外时间开销。

---

## 11. 课程设计报告结构

最终课程设计报告计划采用小论文形式，主要包括以下部分：

```text id="l6vc16"
摘要
1 引言
2 相关工作
3 方法
   3.1 图像分类任务与实验模型
   3.2 SGD 与 Adam 优化算法
   3.3 SAM 优化算法
   3.4 改进方法：Warmup-SAM
4 实验设计
   4.1 数据集
   4.2 实验环境
   4.3 参数设置
   4.4 评价指标
5 实验结果与分析
   5.1 基础优化器对比实验
   5.2 参数调优实验
   5.3 SAM 改进实验
   5.4 训练时间与泛化性能分析
6 结论
参考文献
附录
```

---

## 12. 后续计划

项目后续开发计划如下：

* [ ] 创建项目目录；
* [ ] 编写 `requirements.txt`；
* [ ] 完成 CIFAR-10 数据集加载；
* [ ] 完成 ResNet18 模型构建；
* [ ] 完成 SGD 和 Adam 训练流程；
* [ ] 实现原始 SAM 优化器；
* [ ] 完成 SGD、Adam、SAM 基础对比实验；
* [ ] 完成不同优化器参数调优实验；
* [ ] 实现 Warmup-SAM 改进方法；
* [ ] 完成原始 SAM 与改进 SAM 的对比实验；
* [ ] 绘制实验结果曲线；
* [ ] 整理实验结果表格；
* [ ] 撰写课程设计报告。

---

## 13. 参考文献

[1] Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
Sharpness-Aware Minimization for Efficiently Improving Generalization.
International Conference on Learning Representations, 2021.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[3] Alex Krizhevsky.
Learning Multiple Layers of Features from Tiny Images.
Technical Report, 2009.
