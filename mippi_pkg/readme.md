## Mutation Impact on Protein-Protein Interaction standalone package

### Requirements
* python 3.7+
* tensorflow 2.0+, numpy
* BLAST+ executables (https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
* psiblast search database built from UNIREF90 (https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz)

optional arguments:  
|argument|description|
|:-------|:----------|
|-h, --help|show this help message and exit|
|-i INPUTFILE|input interaction table path|
|-o OUTPUTFILE|output file path|
|-bin BIN|psiblast path, ${bin/psiblast}|
|-db DB|psiblast search database path, recommand UNIREF90|
|-id|with -id, uniprot ID in inputfile, instead of fasta|
|-notmp|with -notmp, delete all intermediate file in /tmp folder, including fasta, PSSM and blast output file|


input file format (detail example and result file in [mippi_pkg/examples](https://github.com/kentergav/MIPPI/tree/master/mippi_pkg/examples)):  
|affected protein reference FASTA|partner protein reference FASTA|mutation annotation|
|:-------|:----------|:------------|

or

|affected protein uniprotAC|partner protein uniprotAC|mutation annotation|
|:-------|:----------|:------------|

&nbsp;
&nbsp;
&nbsp;

eg:  

cd {$mippi_pkg_path}  
(directly use sequence input: )  
python main.py -i ./examples/eg_seq.txt -o ./examples/results_seq1.txt -bin <psiblast_path> -db <psiblast_search_database_path> -notmp  
(use UNIPROT AC as input: )  
python main.py -i ./examples/eg_uniprot.txt -o ./examples/results_uniprot1.txt -bin <psiblast_path> -db <psiblast_search_database_path> -id -notmp  
