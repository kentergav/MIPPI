## Mutation Impact on Protein-Protein Interaction standalone package

### Requirements
* tensorflow 2.0+
* BLAST+ executables (https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
* UNIREF90 (https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz)

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

eg:  

(directly use sequence input: )  
python main.py -i ./examples/eg_seq.txt -o ./examples/results_seq1.txt -bin <psiblast_path> -db <psiblast_search_database_path> -notmp  
(use UNIPROT AC as input: )  
python main.py -i ./examples/eg_uniprot.txt -o ./examples/results_uniprot1.txt -bin <psiblast_path> -db <psiblast_search_database_path> -id -notmp  
