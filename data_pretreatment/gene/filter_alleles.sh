for i in {1..22}
do /home/daiw/miniconda3/envs/algorithm/bin/vcftools --gzvcf /share/pub/likai/myopia/vcf/S8505/S8505.chr"${i}".vcf.gz --min-alleles 2 --max-alleles 2 --recode --recode-INFO-all --out /share/pub/daiw/Multi-Modality-SNPCorrelateImage/data/gene/filtered_alleles_vcf/chr"${i}".vcf
done