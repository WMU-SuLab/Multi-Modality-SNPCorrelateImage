# 数据预处理流程

- 数据格式请参考[数据格式文档](data_format.md)

## disease 标签数据

- 脚本全部都位于`data_pretreatment/label`

### 数据处理

- 第一次处理：将原始数据处理为pandas便于处理的数据格式
    - 运行脚本：`python init.py`
    - 得到的数据文件：`participants_label.csv`
- 第二次处理：区分case和control的数据
    - 运行脚本：`python distibution.py`
    - 得到的数据文件
        - `ADM.csv`:AMD 的case
        - `no_AMD.csv`:AMD 的control
        - `DR.csv`:DR 的case
        - `no_DR.csv`:DR 的control
        - `GLC.csv`:GLC 的case
        - `no_GLC.csv`:GLC 的control
        - `RD.csv`:RD 的case
        - `no_RD.csv`:RD 的control
        - `no_disease.csv`:没有疾病的control
- 第三次处理：根据需要的疾病种类和control的数量，筛选出符合要求的数据，具体细节请查看脚本是如何操作的
    - 运行脚本：`python ukb_screen.py AMD -n 2000`
    - 得到的数据文件：`AMD_2000_${datetime}_screen.csv`

### 处理高度近视数据

- 参考脚本`two_eyes_statistic.py`制定双眼高度近视标准
- 使用脚本添加是否高度近视列，参考脚本`myopia.py`

## gene/snp 数据

- 脚本全部都位于`data_pretreatment/gene`
- ***根据处理SNP数据得到的文件不同，可能需要微调或大调脚本***

### 数据筛选条件

- 为了降低数据的复杂性和不确定性，减少无用数据
- 从青少年高度近视队列中筛选logistic的 OR>1 和 MLMALOCO 的p<0.05的SNP
    - 但是要注意文件中的A1是否和VCF文件中的ALT一致，这会影响到数据的纯合（是0还是2，1是没有影响的）
- 进一步过滤（可选）
    - 单一样本中，如果突变的SNP数量少于某个阈值，就可以认为这个样本的数据质量不好，可以过滤掉这个样本
    - 所有样本中，如果突变的SNP数量少于某个阈值，就可以认为这个SNP的数据质量不好，可以过滤掉这个SNP位点

### 筛选数据

### vcf file

- 本项目目前使用的vcf文件格式是[单染色体文件](data_format.md#single-chromosome-file)
- vcf 文件处理出来之后的表头有一些问题，需要进行处理
    - 首先调用脚本创建表头映射规则：`python data_pretreatment/disease/build_new_vcf_headers.py`
    - 然后调用`bcftools`工具替换表头：`bcftools reheader -s vcf_rename_rules.txt -o new.vcf old.vcf`
- vcf 文件等位基因处理
    - **注意修改脚本中的文件路径和命令路径**
    - 需要重新过滤突变个数（ALT有几种突变）：`nohup sh filter_alleles.sh > filter_alleles.txt 2>&1 &`
    - 过滤完需要压缩：`nohup sh gzip_vcf.sh > gzip_vcf.txt 2>&1 &`
        - 此脚本会自动删除原始文件，如果需要建立索引，需要更改命令使用`-c`
    - 压缩完需要建立索引：`nohup sh index_gzip_vcf.sh > index_gzip_vcf.txt 2>&1 &`

### 风险位点文件

- 调用脚本`filter_snps.py`根据计算值过滤SNP：`python filter_snps.py test.csv`

### participants file

- 需要从原始的vcf文件中提取需要的参与者数据，并修改为自己需要的格式
- 首先提取所需要的参与者，并简化数据
    - 如果样本量不大，调用脚本，例：`python filter_participants.py test.vcf -f participants_ids.txt`
    - 如果文件非常大，这个脚本处理起来会比较慢
        - 建议在做原始数据的时候就进行筛选
            - 比如plink命令自带的筛选功能
            - 由于plink的速度非常慢，所以无法完成拆分参与者的需求，故而产生了后续的流程
        - 使用cyvcf2包进行处理
            - 安装（建议在Linux系统上安装，Windows很难安装）：`pip install cyvcf2`
            - 调用写好的脚本，例：`filter_participants_with_cyvcf2.py test.vcf -f participants_ids.txt`
        - 使用 `bcftools` 从原始数据中筛选出需要的数据
            - 例：`bcftools view -S samples.txt -o new.vcf old.vcf`
    - 如果需要对多个文件进行操作，建议写脚本执行多个程序
        - 原本是脚本做的同时对文件夹内的所有vcf文件进行操作，但是利用操作系统进行进程调动可能效率会更高
- 然后对这些参与者的数据进行转换和规范化：`python regularize test.vcf`
- 接着将提取出来的数据转换为[多人文件的格式](data_format.md#multi-chromosome-file)
    - 原本设计
        - 首先需要拆分文件，每一个参与者会有`VCF文件的SNP总行数/一次读取的行数`个文件
            - 调用脚本，例：`python transpose_split.py test.vcf`
                - 脚本会在`split`文件夹中，按照 participant id 生成多次读取产生的文件，次数取决于一次读取多少行
                - 此外还会保存一个对应的JSON文件`split_group.json`，记录了每个参与者相应的文件路径
            - 多线程的版本：`transpose_split_with_threads.py`
                - 刚开始认为任务的时间大量消耗在存储文件上，但是实际经过测试，发现实际上还是计算消耗的资源多
                - 由于Python的GIL的存在，多线程的效率并不高，所以这个版本的效率也不高，并不需要使用，但是作为记录留存
                - 此外，脚本的操作方法并不巧妙，导致多用了一倍的内存，占用的资源更多了
            - 大内存的版本：`transpose_split_with_large_ram.py`
                - 由于需要对每个参与者的数据进行单独的保存，必然构成新的内存对象，导致内存和多线程版本一样多消耗一倍
                - 根据实际计算，内存的占用大概是原文本数据的40倍，对服务器的内存要求非常高，所以不是很推荐使用
                - 但是这个脚本必然是最快的，如果资源足够，可以使用这个脚本节省时间
        - 最后对所有的文件进行合并，每一个参与者会有一个文件
            - 调用脚本，例：`python transpos_merge.py split_group.json`
            - 这个脚本会根据之前`transpose_split.py`产生的`split_group.json`，将所有的参与者文件合并到各自对应的一个文件中
            - 同时再产生一个所有参与者的总文件，便于后续可能的使用
    - 历程
        - 流程和数据格式在考虑实验性能和使用方便程度的情况下，经过多次更改，最终因为深度学习训练的 dataloader
          加载数据流程的设计，形成了如下设计
        - 经过反复的思考和验证，最后决定将所有人的基因数据单独进行存放，变为一个文件，便于查询和读取
        - 为了大幅度减少存储空间的占用量，将SNP ID（即表头部分）和具体数据分离，否则将多占用一倍的空间
        - 具体见`transpose.py`脚本
- 最终提速
    - 本脚本使用Python原生方法，集成前面的所有流程，一次性完成，并且速度非常快
        - 使用新的加速方法，也写了一个`transpose.py`脚本，代替`split`和`merge`两个流程
    - 调用脚本，例：`python speed_up_full.py input_file outpu_dir -f participants_ids.txt -s 27 --sep \\t`
        - 注意，在不同的情况下，使用`--sep \\t`或者`--sep \t`，这和操作系统、终端类型、ssh客户端、作业提交系统等有关
    - 但是需要注意的是，这个脚本同时打开了大量的文件，所以一定要确保系统没有对此做出限制
        - 可以通过`ulimit -n`查看系统的文件打开限制，默认是1024，如果超出，就要想办法提高
    - **后续所有的改进脚本都应该参照这个脚本进行设计**
- 指定SNP
    - 使用`speed_up_full_with_chosen_snps.py`
      脚本：`python speed_up_full_with_chosen_snps.py /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/ /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/students /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_snps.csv -s 118`
        -
      nohup版本：`nohup python speed_up_full_with_chosen_snps.py /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/ /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/students /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_snps.csv -s 118 > speed_up_full_with_chosen_snps.txt 2>&1 &`
    - 脚本变化
        - 数据文件变为了.gz文件，需要使用`gzip`进行解压处理
        - 单文件变为多文件，即一个文件夹内的所有vcf.gz文件
        - 增加了根据选定SNP进行筛选的功能
        - 增加了纯合判断的功能

## images 数据

- 脚本全部都位于`data_pretreatment/image`
- 现规定图片文件名中必须要有参与者的 id，且放在最前面，图片的格式为 png
- 首选需要筛选出来可用的图片，根据数据集情况有不同的筛选规则，有如下脚本
    - filter.py
    - filter2.py
- 由于图片可能来自各个文件夹，但是 dataset 最终处理的时候是一个文件夹，所以需要进行软连接
    - 为什么不用复制
        - 因为原始的图片数量非常大，复制会占用大量的空间
        - 软连接不会占用空间，但是会有一定的性能损耗
    - `ln -s /path/to/images/* /path/to/dataset/images`
    - 注意要使用绝对路径，否则可能会出现错误
    - 如果文件数量太多，最好使用for循环，否则可能会出现参数过长的错误：`-bash: /usr/bin/ln: Argument list too long`
    - 提供了一个python脚本，`image/ln.py`，可以使用`python ln.py /path/to/images /path/to/dataset/images`来进行软连接
    - 当然如果数据集比较小，也可以使用`copy_files.py`脚本，将图片复制到一个文件夹中
- torchvision 的 transform 模块已经可以对图像进行预处理

## 总体处理

### 数据集不平衡问题

- 由于数据集中的标签不平衡，需要进行欠采样处理，参考`data_pretreatment/label/undersampling.py`

### 归一化和标准化

- 可以在模型，也可以在dataset加载的时候进行处理，具体参考代码`utils/datasets.py`的`GeneNet`模型

### 学籍号和条形码对齐

- 请注意提取出来的各个数据集id是否对应，参考脚本`id_barcode_trans.py`进行修正
- 只能是条形码转学籍号
    - 有的学生重复进行了采样，但是没有重复测序，可能有些条形码数据是没有的，而学籍号是唯一的
    - 有些条形码转不了学籍号，需要从旧测序编号中找到对应的新条形码进行转换
    - 参考脚本`extra_barcodes.py`为`id_barcode_trans.py`的输入数据添加额外数据