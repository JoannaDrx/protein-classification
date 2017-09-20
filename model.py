"""
Model for training
nice description here

Author: Joanna Dreux [joanna.dreux@gmail.com]
"""

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# http://scikit-bio.org/docs/0.1.4/core.alignment.html
# follow iris dataset methodology

import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.Blast.Applications import NcbiblastpCommandline
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pydpi.pypro import GetAAIndex1, GetAAIndex23, CTD, AAComposition, PyPro
import pickle

# cwd
os.chdir('/Users/jdreux/PycharmProjects/protein-classification/')

# Import and eyeball the data

# import the training set
training_set = 'drugpro_training_set.csv'
names = ['id', 'seq', 'label']
ts = pd.read_csv(training_set, header=0)

# check out the df dimensions
print ts.shape
print ts.head(20)
ts.groupby('label').size()
# label
# 0    220
# 1     44

# STEP 1: Sequence homology analysis with BLAST-P
query = 'sg_putative_10000.faa'
subject = 'training_seqs.fasta'
out_blastp = 'blastp_output.tab'
annotation_csv = 'seq_homology_annotation.csv'

# Convert training set to FASTA file format, annotate with label in description
seq_list = []
for id, seq, label in ts.itertuples(index=False):
        seq_record = SeqRecord(Seq(seq, IUPAC.protein), id=id+str("-label:"+str(label)),
                               description="label: "+str(label), name="")
        seq_list.append(seq_record)
# write FASTA
handle = open(subject, "w")
for sequences in seq_list:
    SeqIO.write(sequences, handle, "fasta")

# blastp against the training set for seq homology
blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=1e-7, outfmt=6, out=out_blastp)
print blastp_cline
stdout, stderr = blastp_cline()

# read the results
df = pd.read_table(out_blastp, header=None)
default_outfmt6_cols = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
df.columns = default_outfmt6_cols

# filter for pident >= 90.0
df_filtered = df[df['pident'] >= 90.0]

# positive matches  -- 82 seqs
pos_matches = df_filtered[df_filtered['sseqid'].str.contains('label:1')]
# look at top matches by bitscore
pos_matches.sort_values(by='bitscore', ascending=False, inplace=True)
print pos_matches['qseqid'][:10]

# negative matches
neg_matches = df_filtered[df_filtered['sseqid'].str.contains('label:0')]
neg_matches.sort_values(by='bitscore', ascending=False, inplace=True)
print neg_matches['qseqid'][:10]

# neutral matches
# seqs not represented - annotate as x

# build list of seqs from the whole set
query_list = []
for record in SeqIO.parse(query, "fasta"):
    query_list.append(record.id)

# make an annotation csv
with open(annotation_csv, 'w') as f:
    f.write('seq_id,seq_homology\n')
    for match in set(pos_matches['qseqid']): ## need to check if they are repeated somewhere
        f.write(match +',1\n')
    for match in set(neg_matches['qseqid']):
        f.write(match +',0\n')


# STEP 2: Hidden Markov Models -- domain similarity
out_hmmer = 'out_hmmer.txt'
out_hmmer_training = 'out_hmmer-training-set.txt'
pfam_db = 'Pfam-A.hmm'
threshold = 1e-5

# against training set
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer_training,
                                                                                  pfam_db, subject)
os.system(cmd)

# against ours
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, query)
os.system(cmd)


# look for domains with the lowest e-values and higheset bit scores


# STEP 4: Physicochemical features homology using pydpi

db_dir = './pydpi-indices/'

proindex = GetAAIndex1('KRIW790103', path=db_dir)

for record in SeqIO.parse(subject, "fasta"):
    protein = PyPro()
    protein.ReadProteinSequence(record.id)
    print protein.GetAAComp()
    print protein.GetAAindex1()
    print protein.GetAAindex23(path=db_dir)


moran = protein.GetMoranAuto()
for i in moran:
    print i, moran[i]

protein.GetCTD()



## STEP 5. put it all together and figure it out

# write annotation
# make an annotation csv
annotation_csv = 'annotation.csv'
header = ['seq_id', 'seq_homology', 'domain_homology', 'ml_homology']

with open(annotation_csv, 'w') as f:
    f.write(header)


# STEP 6: Score picks with pickled file

### Evaluation  ###

#DrugPro has agreed to synthesize and test 200 proteins of your choice in their
#assay.  Because DrugPro is concerned about whether they can obtain legal rights
#to well-known therapies for chokeophobia, you will be remunerated for proteins
#that score positively in their assay according to how similar they are to
#existing chokeophobia therapeutics in their training set:

 #   >= 78% sequence identity: $1M per protein
  #  < 78% but >= 45% sequence identity: $2M per protein
   # < 45% sequence identity: $4M per protein

moolah = pickle.load(open('protein_nomination_value.pkl', 'rb'))
print moolah