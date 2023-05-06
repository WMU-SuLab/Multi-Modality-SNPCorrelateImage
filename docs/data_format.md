# 自定义数据格式

## 数据说明

- cohort
    - UKB: almost 80000+ people
    - Undergraduate: almost 7000+ people
- disease/label info table file
    - 一般是csv文件
    - **label文件的列没有固定的格式，根据数据集的需要从文件中提取需要的列作为标签**
- gene file
    - 测序会经过`.fastq`、`.bam`、`.vcf`等格式
    - 本项目使用的是从最终`.vcf`格式中提取的数据
- fundus/eye ground images(left and right)
    - 左右眼需要区分到不同的文件夹

## disease 标签

- format

|   ID    | AMD | DR  | GLC | RD  | OD  | OS  | myopia |
|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:------:|
| 1000072 |  0  |  0  |  0  |  0  |  0  |  0  |   0    |

## Gene file

### raw gene file

- 提取数据方法:``

### pretreated gene file

#### single person file

- 单人文件包含了一个人所有的 SNP 突变相关信息，是中间文件，用于后续的数据处理
- dir path:`/share2/pub/sunhj/sunhj/Data/GeneticImageAI/gene`
    - single person file name format:`{participant_id}_gene.txt`
- format

| SNP ID | Chromosome |  Loci  | Variant | Allele | Count |
|:------:|:----------:|:------:|:-------:|:------:|:-----:|
|  rs1   |    chr1    | 515235 |    G    |  0/1   |   1   |

#### single chromosome file

- 提取的单染色体文件
    - 使用这种格式是因为这是SNP处理的常用方法，代码写起来方便，而上述的单人文件提取起来比较麻烦
- format
    - 6行注释，文件信息
    - 1行表头注释，表头如下
    - 其中如`0/0`这种数据就代表突变情况，`./.`是做了质控之后发现没测到

| CHROM |   POS    |    rsID     | REF | ALT |  QUAL  | FILTER | INFO | FORMAT | Participant ID1 | ...... | Participant IDn |
|:-----:|:--------:|:-----------:|:---:|:---:|:------:|:------:|:----:|:------:|:---------------:|:------:|:---------------:|
|  22   | 16050527 | rs587769434 |  C  |  A  | 511.71 |   .    |  PR  |   GT   |       0/0       | ...... |       0/0       |

#### multi chromosome file

- format 同 single chromosome file
- 合并22个染色体文件，一般不做第23个
    - 也可以在提取的时候就提取所有的染色体

#### final snp files

- format

| Participant ID | SNP ID1 | SNP ID2 | ...... | SNP IDn |
|:--------------:|:-------:|:-------:|:------:|:-------:|
|    1000072     |    0    |    1    | ...... |    2    |

- 无论是前面的任意一种格式，都要处理成这种格式
- 每个人一个文件，再加上一个表头文件
    - 不能合并成一个文件的原因
        - 文件太大，内存无法一次性读取
        - 即使能够读取，查找的过程也比较麻烦
    - 目前只做几十个相关位点，验证效果之后再做全基因组的更多位点

## images

- size:2048 x 1536
- file name format: ID_其他标识符.png
    - 只需要 ID 即可
    - 注意OS、OD的区分
