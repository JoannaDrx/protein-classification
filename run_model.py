
##### Collect Data
query = 'sg_putative_10000.faa'
subject = 'training_seqs.fasta'
out_blastp = 'blastp_output.tab'
annotation_csv = 'seq_homology_annotation.csv'

# blastp against the training set for seq homology
blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=1e-7, outfmt=6, out=out_blastp)
print blastp_cline
stdout, stderr = blastp_cline()

# read the results as above
# against ours
# we build a custom HMM refrence from our three domains of interest obtained at: http://pfam.xfam.org/
# concatenate all three HMM files for these endotoxin families

pfam_db = 'endotoxins-hmm/endotoxins.hmm'
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, query)
os.system(cmd)



# STEP 6: Score picks with pickled file

### Evaluation  ###

#DrugPro has agreed to synthesize and test 200 proteins of your choice in their
#assay.  Because DrugPro is concerned about whether they can obtain legal rights
#to well-known therapies for chokeophobia, you will be remunerated for proteins
#that score positively in their assay according to how similar they are to
#existing chokeophobia therapeutics in their training set:

#   >= 78% sequence identity: $1M per protein
#  < 78% but >= 45% sequence identity: $2M per protein
#  < 45% sequence identity: $4M per protein

moolah = pickle.load(open('protein_nomination_value.pkl', 'rb'))
print moolah