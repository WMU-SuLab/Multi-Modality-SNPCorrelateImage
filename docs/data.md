# Data

## 数据描述

- UKB数据
- SuLab 大学生队列数据

## 数据存放位置

- 眼视光服务器
    - 青少年高度近视队列
        - SNP
            - 带P值：/share2/pub/yaoyh/yaoyh/2Data/1Myopia_public/HM_single_variant_YJ.0916.txt
            - 筛选过的：D:\BaiduSyncdisk\Data\SuLabCohort\gene\filtered_snps.csv
            - 原始VCF：/share2/pub/xingsl/xingsl/WMU_WES/merge/S8505/S8505.chr*.vcf.gz
        - Disease Label
            - D:\BaiduSyncdisk\Data\SuLabCohort\label\students_qc
    - UKB数据
        - 图像数据
            - /share2/pub/biobank/Image/fundus_image_left
            - /share2/pub/biobank/Image/fundus_image_right
        - Disease Label
            - raw file:`/share2/pub/biobank/Image/participants_label.txt`
            - /share2/pub/yaoyh/yaoyh/1Project/4UKBB/participants_label.txt
            - pretreated disease file:`/share2/pub/sunhj/sunhj/Data/GeneticImageAI/participants_label.csv`
        - SNP
            - 原始二进制数据：/share2/pub/biobank/Genotype/Imputation
            - VCF
                - /share2/pub/yaoyh/yaoyh/1Project/4UKBB/image_genetic_genotype/AMD_QCed/
                - /share2/pub/yaoyh/yaoyh/1Project/4UKBB/image_genetic_genotype/DR_QCed/AMD_Control.001.reheader.vcf
- 茶山服务器
    - 大学生队列
        - SNP
            - Raw VCF:/share/pub/likai/myopia/vcf/S8505
            - filtered alleles vcf:/share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf