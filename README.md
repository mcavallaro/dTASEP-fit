# dTASEP-fit

This repository contains software to support M. Cavallaro, Y. Wang, D. Hebenstreit, R. Dutta "Bayesian inference of polymerase dynamics over the exclusion process" (2023) R. Soc. Open Sci. 10: 221469 [doi:10.1098/rsos.221469](https://doi.org/10.1098/rsos.221469) [ArXiv:2109.05100]{https://arxiv.org/abs/2109.05100}

`samplersKappa.py` defines the MCMC sampler. `dWASEP_2.py` and `dWASEP_2_numba.py` define the PDE solver. `stochastic_sim` includes code for stochastic simulations.


If you happen to find any use of this code, please do not forget to cite the paper ;-)



## Preprocess Data

1. Create the bedGraph files.

    *  We consider data available from Gene Expression Omnibus with accession number [GSE117006](https://www.ncbi.nlm.nih.gov/gds/?term=GSE117006). Convert the binary bigWig files to bedGraph using UCSC big* tools:
       ```bash
       # conda install -c bioconda ucsc-bigwigtobedgraph
       for i in $(ls GSM326728*); do bigWigToBedGraph $i $i.bg; done
       ```
       The bedGraph files are a sparse format.

    *  Download the list of genes for the reference genome hg19 and extract their genomic coordinates.
       ```bash
       wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refFlat.txt.gz
       wget https://genome.ucsc.edu/goldenPath/help/hg19.chrom.sizes
       zcat refFlat.txt.gz | awk -v OFS='\t' '{x=$5 - $6; print $3, $5, $6, $1, ( x >= 0 ) ? x : -x , $4}' > gene_coordinates.bg
       ```

    *  Use `bedtools slop` to start the reads upstream of the TSS (where detected PolII is tipycally close to zero).
       ```bash
       bedtools slop -i gene_coordinates.bg -g hg19.chrom.sizes -l 500 -r 0 -s | bedtools sort  > gene_coordinates_slopped.bg
       ```

    *  Remove genes that now overlap by further 500 bases downstream. Remove also pseudo-genes.
   
       ```bash
       bedtools slop -i gene_coordinates_slopped.bg -g hg19.chrom.sizes -r 500 -l -s | bedtools sort | bedtools merge -c 1 -o count | awk ' { if($4 > 1) {print $0} }' > overlapping_genes.bg
       bedtools subtract -a gene_coordinates_slopped.bg -b overlapping_genes.bg -A | awk -v OFS='\t' '{if(($3 - $2) > 2800) {print $0}}'> gene_coordinates_filtered0.bg
       cat gene_coordinates_filtered0.bg | wc -l
       # 5747
       ```
    * Eliminate haplotypes:

      ```bash
      grep -v _hap gene_coordinates_filtered0.bg > gene_coordinates_filtered.bg
      ```


2. Extract the data for the gene and the total reads.
    * The coordinates of the gene ends are known from the reference genome and are
    now stored in `gene_coordinates_filtered.bg`.
      As an example, for the Beta-actin gene (ACTB) the reads from the control experiment:

      ```bash
      awk '{if ($1=="chr7" && $2>5565978 && $3<5570232 ) {print $2 " " $3 " "  $4}}'  GSM3267288_HCT116_M12_control_Spt5_170914_R1.bw.bg > ACTB_control
      ```

      This step must return gene-specific files of the form `ACTB_control`, `ACTB_treatment_time1`, `ACTB_treatment_time2`, etc.
      The `bash` scripts `extract.sh` and `extract_control.sh` perform these operations for bedformat given lists of genes and ordered sequencing data.
      First only check the control file reads, in oder to select the genes with the highest density.
      ```bash
      parallel --gnu -P 20  --colsep '\t' bash extract_control.sh {} < gene_coordinates_filtered.bg
      ```
      The resulting files `gene+chr_control` are still in sparse format (see point 3).



    * Print the average density of PolII. A one-liner is:
      ```bash
      awk 'BEGIN {somma=0} NR==1{init=$1} {somma=somma+$3*($2-$1)} END {print somma/($1-init)}'  A2MP1-chr12_control
      ```
      Sequentially apply to all:
      ```bash
      cd Chip-seq_Trp_Spt5_
      echo -e "gene\ttotal_reads\tgene_length\tread_density" > total_reads.dat
      for f in $(ls *control);
        do awk -v OFS='\t' 'BEGIN {somma=0} NR==1{init=$1} {somma=somma+$3*($2-$1)} END {split(FILENAME, a, "_"); x=init - $2; print a[1], somma, ( x >= 0 ) ? x : -x, somma / (( x >= 0 ) ? x : -x) }'  $f;
      done >> total_reads.dat
      ```
 
      Therefore `total_reads.dat` has columns `gene`, `total reads`, `gene_length`, `read density`.
      ```bash
      head total_reads.dat
      #gene  total_reads gene_length read_density
      #A2MP1-chr12  26581.1 6400  4.15329
      #A3GALT2- 89907.5 15050 5.97392
      #A4GNT-chr3 30944.7 9000  3.4383
      #AADACL4+ 86029.3 23150 3.71617
      mv total_reads.dat ..
      ```

    * Merge `total_reads.dat` and `gene_coordinates_filtered.bg` in order to obtain a bedGraph with read totals and densities.
     ```bash
     python merger.py total_reads.dat gene_coordinates_filtered.bg -o gene_coordinates_filtered_with_density.bg
     head gene_coordinates_filtered_with_density.bg -n 3
     #chr     start   end     gene    length  strand  total_reads     gene_length     read_density    name
     #chr17   73772514        73776360        H3-3B   3346    -       719836.0        3800    189.43099999999998      H3-3B-chr17
     #chr17   80247921        80251190        LINC01970       2769    -       477972.0        3200    149.366 LINC01970-chr17
     ```

    * Select the 1000 genes with the highest density.
     ```bash
     head -n 1000 gene_coordinates_filtered_with_density.bg > gene_coordinates_filtered_with_density_filtered.bg
     ```

    * Extract reads for the other time-course experiments using `extract.sh`.
     ```bash
     parallel --gnu -P 20 --colsep '\t' bash extract.sh {} < gene_coordinates_filtered_with_density_filtered.bg
     ```


3. (Optional) Create matrix files which can be readily used by the MCMC sampler function.

    This short pipeline can be called from a python interactive session:
    ```python
    from dTASEP import SeqUtils
    SeqUtils.process('Data/ACTB')
    ```
    or using the two stand-alone python scripts.

    * Aggregate the data from the time-course experiments to one dense matrix file, where each column correspond to a time-course experiment:

     ```
     python Preprocess.py Data/ACTB
     ```

     This script searches for input files of the form `ACTB_control`, `ACTB_treatment_time1`, `ACTB_treatment_time2`, etc., and creates a matrix file called `ACTB.dat`.
     A file `ACTB_sequence` containing the gene sequence is optionally searched in the same directory and is merged into the output matrix.

     To retrive the list of gene use (thus avoiding to double count all the genes with suffix `control`).
     ```bash
     cd Chip-seq_Trp_Spt5_
     ls *Trp_10 | cut -f1 -d_  > list_of_genes.dat
     mv list_of_genes.dat ..
     ```

     Then:
     ```bash
     parallel --gnu -P 20  python ../Preprocess.py {} < ../list_of_genes.dat
     ```


    * Bin the data:
      ```bash
      # python Bin.py Data/ACTB.dat --bin-size 200
      ls *dat > ../list_of_gene_files.dat
      parallel --gnu -P 20  python ../Bin.py --bin_size 20 {} < ../list_of_gene_files.dat
      ```
      This create matrix files such as called `ACTB-chr7_binned_195.dat`, where the `195` is the number of bins and $195 * 20 = 3900$ is the length of the genes in base-pairs.
      `Bin.py` also defines kernel-smoothing utils. The second argument is optional with default value `200`.



4. From [2], we know that the elongation rate is ~2000 bases / min, so a gene with 195 bins each of 20 bases. We use this to derive the maximum rate to be used in the simulation.


5. Spike-ins normalisation from `fastaq` data. Genomes as fasta files are http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz and http://hgdownload.soe.ucsc.edu/downloads.html.


```bash
  wget http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
  wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
  gzip -d hg18.fa.gz
  gzip -d mm10.fa.gz

  # sratools
  fastq-dump SRR7515455
  fastq-dump SRR7515456
  fastq-dump SRR7515457
  fastq-dump SRR7515458
  fastq-dump SRR7515459

  # hisat2
  sed 's/chr/mchr/g' mm10.fa > mmm10.fa
  cat hg19.fa mmm10.fa > one-big-file.fa
  hisat2-build one-big-file.fa massimo-custom-index
  hisat2 -x massimo-custom-index -U SRR7515455.fastq -S SRR7515455.sam

  # convert sam to bam
  for i in $(ls *sam); do orig=$(echo ${i} | sed -e 's/.sam//'); do
    samtools sort -o ${orig}.bam ${orig}.sam;
  done

  # index bam:
  for f in $(ls *bam); do
    samtools index $f
  done

  # print
  for f in $(ls *bam); do
    echo $f
    echo " spike-in reads:"
    samtools idxstats $f | awk '{print $1" "$3}' | grep mchr | awk 'BEGIN {somma=0} {somma=somma+$2} END {print somma}'
    echo " human reads:"
    samtools idxstats $f | awk '{print $1" "$3}' | grep -v mchr | awk 'BEGIN {somma=0} {somma=somma+$2} END {print somma}'
  done >> spike-in_reads.txt

```



## Find total reads for Kappa prior

Spt5:
```bash
awk 'BEGIN{somma=0}{somma = somma + ($3 -$2) * $4} END {print(somma)}' GSM3267288_HCT116_M12_control_Spt5_170914_R1.bw.bg
```
1.14799e+10


PolII:
```bash
awk 'BEGIN{somma=0}{somma = somma + ($3 -$2) * $4} END {print(somma)}'  GSM3267283_HCT116_control_PolII_160809_R4.bw.bg
```
1.00428e+06


## References

[1] B. Erickson, R.M. Sheridan, M. Cortazar, D.L. Bentley, Dynamic turnover of paused pol II complexes at human promoters, Genes Dev. 32 (2018) 1215–1225. doi:10.1101/gad.316810.118.

[2] I. Jonkers, H. Kwak, J.T. Lis, Genome-wide dynamics of Pol II elongation and its interplay with promoter proximal pausing, chromatin, and exons, Elife. 2014 (2014) 1–25. doi:10.7554/eLife.02407.
