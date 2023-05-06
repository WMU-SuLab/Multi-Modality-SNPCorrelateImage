for i in {1..22}
do /home/daiw/miniconda3/envs/algorithm/bin/tabix -p /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/chr"${i}".vcf.recode.vcf /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/chr"${i}".vcf.recode.vcf.gz
done