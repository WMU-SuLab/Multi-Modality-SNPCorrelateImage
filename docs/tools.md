# 工具

记录一下常用的基因数据处理工具

## vcftools

- 安装：`conda install -c bioconda vcftools`
- 查看帮助手册：`man /share/apps/vcftools/share/man/man1/vcftools.1`
- 使用
    - 筛选alleles：`vcftools --vcf input.vcf --min-alleles 2 --max-alleles 2 --recode --recode-INFO-all --out output.vcf`
      -
      如果输入的是vcf.gz: `vcftools --gzvcf input.vcf.gz --min-alleles 2 --max-alleles 2 --recode --recode-INFO-all --out output.vcf`
        - vfctools 无法同时处理多个文件，需要写一个shell脚本，循环处理

## bgzip & tabix

- 安装：`conda install -c bioconda tabix`

### bgzip

- 文档：<http://www.htslib.org/doc/bgzip.html>
- 使用
    - 压缩，不保留原始文件：`bgzip *.vcf`
    - 压缩，使用管道，保留原始文件：`bgzip -c input.vcf > input.vcf.gz`

### tabix

- 使用
    - 建立索引：`tabix -p *.vcf *.vcf.gz`

## bcftools

- 安装：`conda install -c bioconda bcftools`

## plink

- 安装：`conda install -c bioconda plink`

