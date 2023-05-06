# 训练流程

## 数据预处理

- 数据预处理主要是基因和标签数据的处理，根据UKB和大学生队列数据的情况，稍微调整处理方法
- 基因的处理方法已经基本定下来了，主要参考[数据预处理文档](data_pretreatment.md)

## 划分数据集

- 请注意
    - 原始数据集最好是分为左右眼，因为是区域关联，不同方向会影响模型的效果
    - 划分数据集和图像数据预处理中的操作虽然看起来很像，但是作用是不同的
        - 划分数据集是已经选择好了哪些数据来训练，划分出训练集、验证集和测试集
        - 图像数据预处理是对图像进行一些处理，并且从图像库中提取可使用的数据
        - 是否可使用是根据图像质量、有无对应标签和基因数据等来判断的
- 使用`dataset.py`模块
    - 使用方法：`python divide_dataset.py label_data_path gene_data_dir_path image_data_dir_path`
    - 默认的可选参数相当于：`--train_ratio 7 --valid_ratio 3 --strategy train_valid`
        - 根据需要调整数据集的比例
    - 可以更改数据集存放的文件夹路径：`--dataset_divide_dir xxx_dir`
    - 其余需要更改代码
        - 列的数据类型 dtype
        - id 列的字段名称
- 示例：
    - `python divide_dataset.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/label/ftd_myopia_left.csv /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/gene/students/ /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/image/students_qc/ftd_left/ --dataset_divide_dir /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/ --label_data_id_field_name 学籍号`

## 加载数据集

- 使用`dataloader.py`中的模块
    - SNPImageDataset
        - 注意更改label的ID和Value字段名称

## 模型训练

- weights 和 checkpoints 的区别
    - weights是模型参数，可以直接使用 load_state_dict() 方法加载
    - checkpoints 包含了其他参数，如epoch、optimizer等
    - 本项目所有变量名称都符合这个原则，注意区分
- 前提是使用`divide_dataset.py`模块划分好数据集
- 使用`train.py`模块
- 示例
    - `python train.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230411103345/ 12275 --pretrain_image_feature_checkpoint_path /data/home/sunhj/Multi-Modality-SNPCorrelateImage/weights/convnext_tiny_1k_224_ema.pth`
    - `python train.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230411170537/ 12275 --epochs 100 --batch_size 64`
