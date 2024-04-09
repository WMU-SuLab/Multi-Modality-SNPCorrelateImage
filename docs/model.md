# Model

- 代码都在`models`
- 将所需要训练的的模型在`__init__.py`对应类中标明
    - `SNP.py`:基因模型的实现
    - `Image.py`:图像模型的实现
    - `MultiModal.py`:多模态模型的实现
    - `Attention.py`:多种注意力机制实现
    - `RETFound.py`:Nature 的预训练模型原始代码

## 基因模型 SNPNet

- 基本的MLP
- 基于位置编码的改进模型

## 图像模型 ImageNet

- ConvNeXt
- RETFound：基于ViT的预训练模型

## 多模态模型 SNPImageNet

- 基因模型和图像模型分别提取特征
- 融合方法
    - Concat
    - 各种Attention
    - TransformerEncoder
    - ……
