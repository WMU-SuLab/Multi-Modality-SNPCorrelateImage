# 训练流程

## 初始化文件夹

- 使用`init_dirs.py`初始化项目

## 数据预处理

- 处理方法主要参考[数据预处理文档](data_pretreatment.md)
- 数据预处理主要是基因和标签数据的处理，根据UKB和大学生队列数据的情况，稍微调整处理方法
- 图像方面，首先通过人工筛选图片，训练ConvNeXt模型来筛选出一部分高质量图片
  - 接下来的人工再筛选是可选方法
  - 经过实验，人工再筛选的图像模型似乎并不更好

### 根据频率预筛选位点

- 参考`data_pretreatment/gene/participants_snps_filter_with_frequency.py`
- 选取策略，根据不同的频率范围预筛选位点

## GWIS

- 使用分组重要性评分挑选可能相关的基因，再将以基因为单位的SNP作为模型的输入
- 首先使用`data_pretreatment/gene/participants_snps_to_gene_regions.py`将SNP数据转化为基因分组的数据
  - 其中基因分组数据使用`data/gene/gencode.v19.annotation.gtf.gene_regions.json`，由`gencode.v19.annotation.gtf`文件计算而来
  - 如果发现还需要再过滤一次SNP，使用`data_pretreatment/gene/participants_gene_regions_filter_with_chosen_snps.py`
  - 另一种方法:`data_pretreatment/gene/vcf_gene_regions.py`和`data_pretreatment/gene/vcf_filter_snps_with_gene_regions.py`是直接从VCf文件中提取基因分组数据
- 接着运行`gwis.py`，得到记录
- 最后使用`data_pretreatment/gene/participants_gene_regions_filter_snps_with_selected_gene.py`选取需要的位点，重新组织为原始数据

## 划分数据集

- 请注意
    - 原始数据集最好是分为左右眼，因为是区域关联，不同方向会影响模型的效果
    - 划分数据集和图像数据预处理中的操作虽然看起来很像，但是作用是不同的
        - 划分数据集是已经选择好了哪些数据来训练，划分出训练集、验证集和测试集
        - 图像数据预处理是对图像进行一些处理，并且从图像库中提取可使用的数据
        - 是否可使用是根据图像质量、有无对应标签和基因数据等来判断的
- 使用`divide_dataset.py`模块
    - 使用方法：`python divide_dataset.py label_data_path gene_data_dir_path image_data_dir_path`
    - 默认的可选参数相当于：`--train_ratio 7 --valid_ratio 3 --strategy train_valid`
        - 根据需要调整数据集的比例
    - 可以更改数据集存放的文件夹路径：`--dataset_divide_dir xxx_dir`
    - 应用在其他数据集上需要更改代码
        - 列的数据类型 dtype
        - id 列的字段名称
        - ……
    - 其余参数见代码，或者`python divide_dataset.py -h`
- 示例：`python divide_dataset.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/label/ftd_myopia_left.csv /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/gene/students/ /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/image/students_qc/ftd_left/ --dataset_divide_dir /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/ --label_data_id_field_name 学籍号`

## 基因数据集归一化（可选）

- 使用`data_pretreatment/gene/count_gene_freq.py`获得当前划分数据集中训练集的频率
- 用于将原始SNP数据转为SNP在人群中的频率，此模块根据实验结果发现影响很小，可以跳过
- 示例：`python count_gene_freq.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230426211048/train/gene`

## 模型测试

- 参考`test_model.py`先看看模型能不能跑

## 模型训练

- weights 和 checkpoints 的区别
    - weights是模型参数，可以直接使用 load_state_dict() 方法加载
    - checkpoints 包含了其他参数，如epoch、optimizer等
    - 本项目所有变量名称都符合这个原则，注意区分
    - 现在基本已经把weights用checkpoints代替，不保存两份，虽然加载慢一点，但是空间占用少了很多
- 前提是使用`divide_dataset.py`模块划分好数据集
- 使用`train.py`模块
    - 输入模型名称
    - 输入数据集路径，即划分数据集得到的文件夹路径
    - 输入基因数据长度，可使用`data_pretreatment/gene/count_gene_num.py`得到
    - 输入基因频率文件路径（可选）
    - ……
    - 其他参数见代码
- 示例
    - `python train.py SNPImageNet /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230426211048 --snp_numbers 21213 --gene_freq_file_path /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230426211048/train/gene/freq.json --epochs 100 --batch_size 32` 
    - `python .\train.py SNPImageNet F:\sunhj\Multi-Modality-SNPCorrelateImage\data\divide\20230626103943 --pretrain_image_feature_checkpoint_path .\weights\convnext_tiny_1k_224_ema.pth --snp_numbers 25190 --epochs 100 --batch_size 32`
- `train_multi.py`：多显卡训练，长时间不用了，可能需要更改才能运行


### 可调整策略

- 模型的初始化权重分布
- 图像的transformer处理策略
- 各种学习率、batch size等
- 是否使用预训练（未完善，似乎有问题，续改进）


## 测试模型

- 参考`test.py`测试数据集
- 示例
    - `python test.py /pub/sunhj/data/Multi-Modality-SNPCorrelateImage/divide/20230411170537/ /data/home/sunhj/Multi-Modality-SNPCorrelateImage/weights/20230411194940/epoch_10_model_wts.pth  12275 /home/sunhj/Multi-Modality-SNPCorrelateImage/logs/20230411194940`
- 参考`predict.py`预测某个数据

## 权重计算

- 参考`visualize*.py`代码，不同之处在于单个数据计算、数据集计算和自定义权重计算方式

## 再筛选位点

- 使用`data_pretreatment/gene/select_snps_by_weight.py`选择位点
- 使用`data_pretreatment/gene/participants_snps_filter_with_chosen_snps.py`过滤出位点数据

## 清理训练失败的文件夹

- 先删除log文件夹
- 调用`clear_dirs.py`
