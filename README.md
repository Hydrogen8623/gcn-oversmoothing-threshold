# GCN Oversmoothing Threshold

**本项目验证：在合成图上，GCN 的节点分类准确率随层数加深在哪个临界层数因过平滑而显著下降，以及该临界层数是否随图的社区数变化。**

## 研究问题

GCN 的层数并非越深越好。随着层数增加，节点表示趋于同质化，节点分类性能下降，这一现象称为过平滑。已有工作表明，多数 GCN 在 2–3 层时性能达到峰值，更深层网络性能停滞或退化（Li et al., 2018; Oono & Suzuki, 2020）。然而，过平滑的临界层数是否随图的社区结构变化，尚缺乏系统性的对照实验。

本项目在受控条件下改变 GCN 层数与图的社区数，测量节点分类准确率，定位过平滑发生的临界层数，并检验该临界点与社区数之间是否存在单调或非单调关系。

## 方法

- 数据集：合成图，社区数取 {2, 4, 8, 16}，节点特征维度与边密度固定。
- 模型：标准 GCN（Kipf & Welling, 2017），层数取 1–8。
- 对照设计：固定随机种子集合，每个（层数，社区数）组合重复多次，报告均值与标准差。
- 指标：节点分类准确率；辅以节点表示的平均 pairwise 余弦相似度，用于直接观察过平滑程度。
- 验证数据集：Cora（Planetoid 划分），用于验证实现的正确性，并检验合成图结论是否在真实图上成立。

## 运行训练

使用 Cora 数据集配置运行：

```bash
python train.py --config configs/single/cora.yaml
```

## 结果

### Cora Baseline

| 参数 | 数值 |
|---|---:|
| Dataset | Cora |
| GCN Layers | 2 |
| Hidden Dimension | 16 |
| Epochs | 200 |
| Best Validation Accuracy | 0.802 |
| Test Accuracy | 0.812 |

实验结果来自两次独立运行，两次运行取得一致的测试准确率。

当前实验结果仅作为项目 Baseline，后续将进一步研究 GCN 层数增加时的过平滑现象。

## 复现


本项目已实现 GCN 基础训练管线，支持在 Cora 数据集上进行节点分类实验。

### 1. 环境要求

- Python 3.11
- 依赖见 `requirements.txt`

### 2. 安装依赖

建议使用 Conda 创建独立环境：

```bash
conda create -n gcn-oversmoothing-threshold python=3.11 -y
conda activate gcn-oversmoothing-threshold
pip install -r requirements.txt
```

### 3. 运行实验

使用 Cora 数据集配置运行训练：

```bash
python train.py --config configs/single/cora.yaml
```

具体配置参数以 `configs/single/cora.yaml` 为准。

### 4. 实验结果

当前 Cora Baseline 实验配置：

| 参数 | 数值 |
|---|---:|
| 模型 | GCN |
| 数据集 | Cora |
| GCN 层数 | 2 |
| Hidden Dimension | 16 |
| Epochs | 200 |
| Best Validation Accuracy | 0.802 |
| Test Accuracy | 0.812 |

在相同配置下进行两次独立运行，均获得 81.2% 的测试准确率。

### 5. 实验输出

训练完成后，实验日志及结果保存在 `outputs/` 目录下。

本项目使用验证集选择最佳模型，并仅在最终评估阶段使用测试集。

## 局限

待补。

## 项目状态

- [x] GCN 基础模型
- [x] Cora 数据集训练
- [x] 验证集模型选择
- [x] 测试集最终评估
- [x] 基础测试
- [ ] 多层 GCN 过平滑实验
- [ ] 合成数据集实验
- [ ] 过平滑阈值分析