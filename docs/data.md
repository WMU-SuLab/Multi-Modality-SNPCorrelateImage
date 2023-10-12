# Data

记录的是服务器上原始文件位置，非处理后的数据

## 数据描述

- UKB 数据
- SuLab 大学生队列数据
  - 图像数据
    - mean: mean_b:40.587787595185105,mean_g:56.897545523381226,mean_r:80.04950144849344，即0.1585,0.2222,0.3127
    - std: sd_b:9.247330111977062,sd_g:10.289160282303381,sd_r:11.616102441651151，即0.0361,0.0402,0.0454



## 数据存放位置

- 眼视光服务器
    - 大学生青少年高度近视队列
        - 图像数据
            - /share2/pub/yangjy/yangjy/image/Data/
        - 基因数据
            - Summary
                - 原始
                    - /share2/pub/yaoyh/yaoyh/2Data/1Myopia_public/HM_single_variant_YJ.0916.txt
                    - D:\BaiduSyncdisk\Data\SuLabCohort\gene\HM_single_variant_YJ.0916.txt
                - 筛选过的：D:\BaiduSyncdisk\Data\Multi-Modality-SNPCorrelateImage\gene\SuLabCohort
            - VCF：/share2/pub/xingsl/xingsl/WMU_WES/merge/S8505/S8505.chr*.vcf.gz
        - Disease Label
            - D:\BaiduSyncdisk\Data\SuLabCohort\label\students_qc
    - UKB数据
        - 图像数据
            - /share2/pub/biobank/Image/fundus_image_left
            - /share2/pub/biobank/Image/fundus_image_right
        - 基因数据
            - 原始二进制数据：/share2/pub/biobank/Genotype/Imputation
            - Summary
            - VCF
                - AMD：/share2/pub/yaoyh/yaoyh/1Project/4UKBB/image_genetic_genotype/AMD_QCed/AMD_Control.001.reheader.vcf
                - GLC：/share/pub/xuezb/biobank/GWAS/GLC/data_70359/fastGWA/GLC.fastGWA
        - Disease Label
            - 正博师兄
                - raw file:`/share2/pub/biobank/Image/participants_label.txt`
                    - /share2/pub/yaoyh/yaoyh/1Project/4UKBB/participants_label.txt
                - pretreated disease file:`D:\BaiduSyncdisk\Data\UKB\label\participants_label.csv`
            - 调查问卷：
    - 眼视光医院
        - 图像数据
            - /share2/pub/yangzj/yangzj/pycharm/data/hospital_image/Data
- 茶山服务器
    - 大学生青少年高度近视队列
        - 基因数据
            - Summary
                - 连续性状：/share/pub/yuanj/project/OphthWES/ExWAS/S8114_final_fastgwa_SE1.fastGWA
            - Raw VCF:/share/pub/likai/myopia/vcf/S8505
            - filtered alleles vcf:/share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf
