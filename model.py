"""
Model for training
nice description here

@author: Joanna Dreux
joanna.dreux@gmail.com
"""

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://www.datacamp.com/community/tutorials/machine-learning-python#gs.AsFJLjQ
# https://github.com/mhorlbeck/CRISPRiaDesign/blob/master/Library_design_walkthrough.md
# https://pypi.python.org/pypi/pydpi/1.0
# http://scikit-bio.org/docs/0.1.4/core.alignment.html
# follow iris dataset methodology

import pandas as pd
import sklearn
from Bio import SeqIO, SearchIO, Alphabet
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.Blast.Applications import NcbiblastpCommandline

# STEP 1: Import and eyeball the data

# import the training set
training_set = '/Users/joanna/PycharmProjects/second_genome/drugpro_training_set.csv'
names = ['id', 'seq', 'label']
ts = pd.read_csv(training_set, header=0)

# check out the data dimensions
print ts.shape
print ts.head(20)
ts.groupby('label').size()
# numbers here
#label
#0    220
#1     44

# just from eyeballing the data, there is high sequence homology between the 1s
# convert training set to correct file format
# write label 1s as fasta file
seq_list = []
for id, seq, label in ts.itertuples(index=False):
        pos_seq = SeqRecord(Seq(seq, Alphabet.IUPAC.protein), id=id+str("--label:"+str(label)), description="", name="")
        seq_list.append(pos_seq)

# write FASTA
handle = open('/Users/joanna/PycharmProjects/second_genome/training_seqs.fasta',"w")
for sequences in seq_list:
    SeqIO.write(sequences, handle, "fasta")

# STEP 2: Sequence homology analysis with BLAST-P
# blast the 1s against each other

query = '/Users/joanna/PycharmProjects/second_genome/sg_putative_10000.faa'
subject = '/Users/joanna/PycharmProjects/second_genome/training_seqs.fasta'
out_blastp = '/Users/joanna/PycharmProjects/second_genome/align_to_1.tab'

blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=1e-7, outfmt=6, out=out_blastp)
print blastp_cline
stdout, stderr = blastp_cline()

# read the results
df = pd.read_table(out_blastp, header=None)
default_outfmt6_cols = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
df.columns = default_outfmt6_cols

# filter for pident >= 90.0
df_filtered = df[df['pident'] >= 90.0]


# look at top matches
# sort by bitscore
df_filtered.sort_values(by='bitscore', ascending=False, inplace=True)
# df_filtered is now sorted by bitscore in descending order
top10_query_ids = df_filtered['qseqid'][0:10]

# positive matches
pos_matches = df_filtered[df_filtered['sseqid'].str.contains('label:1')]

# negative matches
neg_matches = df_filtered[df_filtered['sseqid'].str.contains('label:0')]

# neutral matches
# seqs not represented - annotate as x

query_list = []
for record in SeqIO.parse(query, "fasta"):
    query_list.append(record.id)

# make an annotation csv
annotation_csv = '/Users/joanna/PycharmProjects/second_genome/seq_homology_annotation.csv'
header = ['seq_id', 'seq_homology']

with open(annotation_csv, 'w') as f:
    f.write(header)
    for match in pos_matches


# 2. Hidden Markov Models -- domain similarity

#In addition to sequence identity, hidden-markov models are well known to detect
#"remote hology", that is, a functional relationship between two proteins whose
#sequence identity may be very low.  The most-often used tool for scanning
#protein sequences against a know set of HMMs is hmmer:

#  > HMMER
#    http://hmmer.org/

#A typical source of HMMs with which to annotate proteins using HMMER is the
# Pfam database of protein domains:    http://pfam.xfam.org/

# 3. Machine Learning to classify based on primary sequence

# machine learning models trained on physicochemical features to classify proteins of unknown function from their
# primary sequence.  There is no such thing as "the" set of physicochemcial
# features for a particular protein.  The set is effectively infinite.  However,
# there are several resources for calculating a fairly comprehensive set of
# physicochemcial features from a set of sequences.  They are:

#  > ProFEAT (webserver)
#    http://bidd2.nus.edu.sg/cgi-bin/profeat2016/main.cgi

#  > ProtrWeb (R)
#    http://protrweb.scbdd.com/

#  > Pydpi (Python)
#    https://pypi.python.org/pypi/pydpi/1.0

# 4. Extras

# numeric components
# length, AA content,

from sklearn.cluster import KMeans

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_


# add whatever features you deem necissary from the
# primary sequences to accomplish the classification task, but these should
# provide a sufficient start.

#The way I would attack the problem, in general, is to leverage statistical analysis like Principal Component
# Analysis or Ordinary Least Squares to help determine what attributes within these protein sequences are best
# suited to classify proteins as Neuropeptide hormone precursors.

#In order to do that, you'll have to convert the protein sequences into numeric data, but I believe some work
# has already been done in that regard using formulas leveraged in Amino Acid PCA.

#See these two links: http://www.ncbi.nlm.nih.gov/pubmed/24496727

#http://www.ncbi.nlm.nih.gov/pubmed/16615809

#Once that work has been done, I would attempt to classify using the entire dataset and a reinforcement
# learning algorithm, like Naive Bayes while slimming down the data into that which PCA has identified as important.


# write annotation
# make an annotation csv
annotation_csv = '/Users/joanna/PycharmProjects/second_genome/annotation.csv'
header = ['seq_id', 'seq_homology', 'domain_homology', 'ml_homology']

with open(annotation_csv, 'w') as f:
    f.write(header)


### Evaluation  ###

#DrugPro has agreed to synthesize and test 200 proteins of your choice in their
#assay.  Because DrugPro is concerned about whether they can obtain legal rights
#to well-known therapies for chokeophobia, you will be remunerated for proteins
#that score positively in their assay according to how similar they are to
#existing chokeophobia therapeutics in their training set:

 #   >= 78% sequence identity: $1M per protein
  #  < 78% but >= 45% sequence identity: $2M per protein
   # < 45% sequence identity: $4M per protein

#Normally it requires several months to synthesize and test a protein, so we
#will be accelerating that process with the following validation file:

#  > protein_nomination_value.pkl

#Once you are done training your classifier and applying it to the putative
#set of 10000 proteins, use this pickled dictionary to report the monetary
#value returned by your Top 200 nominations.  Obviously you could pretty
#easily use this dictionary to just go and search for high-valued proteins
#directly in the set of 10000, but that wouldn't make for a very good presentation.