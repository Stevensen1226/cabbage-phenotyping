# PointGroup 对比实验环境搭建指南

PointGroup 依赖于 3D 稀疏卷积库 `spconv` 以及自定义的 CUDA 算子。为避免与当前 `cabbage_pheno` 项目产生依赖冲突，强烈建议为 PointGroup 创建一个**独立的 Conda 虚拟环境**。

## 1. 创建虚拟环境
建议使用 Python 3.8（与多数 PointGroup 实现兼容性最好）：
```bash
conda create -n pointgroup python=3.8
conda activate pointgroup
```

## 2. 安装 PyTorch
请根据您的显卡 CUDA 版本安装对应的 PyTorch（这里以 CUDA 11.8 为例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> 如果您的 CUDA 版本不是 11.8，请前往 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 获取对应的安装命令。

## 3. 安装 spconv (Sparse Convolution)
现代的 `spconv-cu11x` (2.x版本) 已经提供了预编译包，免去了冗长的编译过程：
```bash
# 请将 cu118 替换为您的实际 CUDA 版本，可选的有 cu113, cu114, cu116, cu117, cu118 等
pip install spconv-cu118
```

## 4. 获取 PointGroup 源码
您可以克隆原始的 PointGroup 仓库或使用基于现代 PyTorch 的重构版：
```bash
git clone https://github.com/Jia-Research-Lab/PointGroup.git
cd PointGroup
```

## 5. 编译 PointGroup 核心算子
这一步非常关键，PointGroup 的聚类 (BFS) 等核心操作使用 C++/CUDA 编写：
```bash
# 安装必要的编译组件
pip install ninja pybind11

# 编译算子
cd pointgroup_ops
python setup.py develop
cd ..
```

## 6. 安装另外的依赖包
```bash
pip install tensorboardX pyyaml scipy scikit-learn tqdm plyfile numba
```

## 7. 运行准备
1. 利用 `tools/prepare_pointgroup_data.py` 将甘蓝数据转换为 PointGroup 格式 (\`.pth\`)。
2. 配置 PointGroup 的 `config/` 文件，将其 `classes` 设置为 2 (背景 + 甘蓝)。
3. 开始训练:
   ```bash
   python train.py --config config/pointgroup_cabbage.yaml
   ```
