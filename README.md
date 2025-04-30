# MicroTiClassifier

MicroTiClassifier 是一个基于深度学习的图像分类实验项目。

基于预训练的 VGG16 模型，使用 5 类共 825 张原始图片进行迁移学习，实现对钛合金微观组织图像的分类。

## 项目结构

|模块|功能|详细|
|---|---|---|
|`config`|配置文件|文件路径、训练参数|
|`data`|数据集|无|
|`log`|数据处理日志|运行时生成|
|`output`|模型训练输出|运行时生成|
|`script`|可视化分析脚本|网格图、训练日志、混淆矩阵、PCA 降维、PR 曲线、分类报告、ROC 曲线、数据集统计、t-SNE 降维、UMAP 降维|
|`source`|模型相关脚本|模型导出、类激活热力图、模型预测、语义特征提取、模型训练|
|`util`|数据处理脚本|原始数据集分析、数据增强、图像文件筛选、数据集划分、数据集统计分析|

## 实验环境

|OS|Python|PyTorch|CUDA|
|:---:|:---:|:---:|:---:|
|Windows 11 24h2|3.11|2.3.1|-|
|Ubuntu 22.04 LTS|3.11|2.3.1|12.1|

完整依赖清单：[`pyproject.toml`](pyproject.toml)。

## 快速开始

1. 克隆

```bash
git clone https://github.com/huang2fire/MicroTiClassifier.git
```

2. 配置环境

- uv(**recommend**)

> 安装 [uv](https://github.com/astral-sh/uv)

```bash
cd ./MicroTiClassifier
uv sync
```

- conda

> 安装 [miniforge](https://github.com/conda-forge/miniforge)

```bash
conda create -n MicroTiClassifier python=3.11
conda activate MicroTiClassifier
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers pandas scikit-learn umap-learn matplotlib seaborn jupyterlab tqdm grad-cam onnx
```

3. 数据集

暂不开放。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
