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
pfam_db = '/Users/jdreux/Desktop/hmm-db/Pfam-A.hmm'
threshold = 1e-5

# against training set, find main domains
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer_training,
                                                                                  pfam_db, subject)
os.system(cmd)

# read the results, weird output format
df2 = pd.read_table(out_hmmer_training, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                    usecols=[0,1,3,5,6,7,11,12,13])
header = 'target acc qname qlen e-value score cond-e-value ind-e-value domain-score'.strip().split(' ')
df2.columns = header

# do all valids have one motif in common
pos_matches = df2[df2['qname'].str.contains('label:1')]
len(set(pos_matches['qname']))  # check they're all represented

# limited common motifs
motifs_of_interest = set(pos_matches['acc'])
# does each query have all 3?
set(pos_matches['qname'].value_counts())  # yes // make sure no duplicates

# for each find out the parameters
# delta endotoxin C
endo_C = pos_matches[pos_matches['acc'] == 'PF03944.13']
endo_C.describe()
#             qlen       e-value       score  cond-e-value   ind-e-value    domain-score
# count   44.000000  4.400000e+01   44.000000  4.400000e+01  4.400000e+01   44
# mean   719.409091  4.867906e-42  157.386364  2.318483e-45  1.320950e-41   156.115909
# std      5.163910  3.164621e-41    4.672661  1.506957e-44  8.589714e-41   4.676509
# min    702.000000  2.000000e-47  138.300000  9.400000e-51  5.300000e-47   136.900000
# 25%    719.000000  8.500000e-47  158.200000  4.300000e-50  2.400000e-46   156.975000
# 50%    719.000000  9.650000e-47  158.800000  4.300000e-50  2.400000e-46   157.500000
# 75%    719.000000  1.500000e-46  159.000000  6.350000e-50  3.575000e-46   157.500000
# max    748.000000  2.100000e-40  161.000000  1.000000e-43  5.700000e-40   159.700000

# delta endotoxin M
endo_M = pos_matches[pos_matches['acc'] == 'PF00555.18']
endo_M.describe()
#             qlen       e-value       score  cond-e-value   ind-e-value   domain-score
# count   44.000000  4.400000e+01   44.000000  4.400000e+01  4.400000e+01  44
# mean   719.409091  6.781977e-61  208.143182  1.244841e-64  6.781977e-61   208
# std      5.163910  2.565063e-60    2.250106  4.680958e-64  2.565063e-60   2.25
# min    702.000000  7.700000e-63  201.200000  1.400000e-66  7.700000e-63   201.2
# 25%    719.000000  2.675000e-62  207.400000  4.775000e-66  2.675000e-62   207
# 50%    719.000000  2.200000e-61  207.400000  4.000000e-65  2.200000e-61   207
# 75%    719.000000  2.225000e-61  210.400000  4.000000e-65  2.225000e-61   210
# max    748.000000  1.700000e-59  212.200000  3.100000e-63  1.700000e-59   212


# delta endotoxin N
endo_N = pos_matches[pos_matches['acc'] == 'PF03945.13']
endo_N.describe()
#             qlen       e-value       score  cond-e-value   ind-e-value    domain-score
#count   44.000000  4.400000e+01   44.000000  4.400000e+01  4.400000e+01     44
#mean   719.409091  1.500004e-23  102.343182  5.000013e-27  2.863643e-23     101
#std      5.163910  6.953332e-23    5.523750  2.317777e-26  1.327454e-22     5.52
#min    702.000000  2.100000e-31   79.200000  7.200000e-35  4.000000e-31     78.3
#25%    719.000000  8.250000e-30  102.800000  2.875000e-33  1.550000e-29     101
##50%    719.000000  1.200000e-29  103.500000  4.100000e-33  2.300000e-29    102
#75%    719.000000  2.025000e-29  104.050000  7.150000e-33  4.025000e-29     103
# max    748.000000  3.300000e-22  109.300000  1.100000e-25  6.300000e-22    108


# neg matches
neg_matches = df2[df2['qname'].str.contains('label:0')]
neg_matches = neg_matches[neg_matches['acc'].str.contains('PF03945.13|PF00555.18|PF03944.13')]

# does each query have all 3?
set(neg_matches['qname'].value_counts())  # they can have 1,2,3 or 4 of the relevant domains


# focus especially on those that have 3 // checkout those that have 4


# for each find out the parameters
# delta endotoxin C
endo_Cneg = neg_matches[neg_matches['acc'] == 'PF03944.13']
endo_Cneg.describe()
#              qlen       e-value       score  cond-e-value   ind-e-value  domain-score
#count   181.000000  1.810000e+02  181.000000  1.810000e+02  1.810000e+02   181
#mean    949.066298  2.426051e-08   97.794475  2.958796e-12  2.495475e-08   96.7
#std     268.113359  2.016720e-07   34.493765  2.346067e-11  2.019568e-07   34
#min     622.000000  2.900000e-49   28.200000  6.200000e-50  2.600000e-46   28
#25%     664.000000  1.900000e-36   74.600000  7.500000e-40  3.700000e-36   74
#50%    1121.000000  5.700000e-31  107.700000  5.600000e-34  2.400000e-30   105
#75%    1174.000000  9.200000e-21  125.400000  2.800000e-24  9.200000e-21   124
# max    1522.000000  1.900000e-06  167.000000  2.200000e-10  1.900000e-06   157

endo_Mneg = neg_matches[neg_matches['acc'] == 'PF00555.18']
endo_Mneg.describe()
#              qlen       e-value       score  cond-e-value   ind-e-value  domain-score
# count   133.000000  1.330000e+02  133.000000  1.330000e+02  1.330000e+02   133
# mean   1007.857143  5.302460e-06  115.936842  7.747850e-09  2.005093e-05   115
# std     248.578173  5.345609e-05   49.999248  8.672744e-08  2.171116e-04   50.55
# min     644.000000  8.100000e-62   19.800000  3.500000e-65  2.000000e-61   17.8
# 25%     687.000000  7.200000e-45   84.300000  2.200000e-48  7.200000e-45   83.4
# 50%    1150.000000  1.500000e-32  113.200000  3.800000e-36  1.500000e-32   113.2
# 75%    1180.000000  1.100000e-23  153.500000  3.700000e-27  2.000000e-23   153.5
# max    1522.000000  6.100000e-04  208.800000  1.000000e-06  2.500000e-03   207.6


endo_Nneg = neg_matches[neg_matches['acc'] == 'PF03945.13']
endo_Nneg.describe()
#              qlen       e-value       score  cond-e-value   ind-e-value  domain-score
##count   195.000000  1.950000e+02  195.000000  1.950000e+02  1.950000e+02   195
#mean    949.420513  1.656490e-07   79.714359  1.416181e-07  1.020671e-03   75
#std     270.116828  2.149497e-06   21.636269  8.539274e-07  6.273761e-03   26
#min     551.000000  3.300000e-35   23.800000  2.200000e-38  9.100000e-35   12
#25%     665.000000  8.050000e-28   63.550000  4.500000e-31  2.500000e-27   61
#50%    1117.000000  5.900000e-24   84.900000  1.300000e-26  5.400000e-23   81
#75%    1180.000000  2.000000e-17   97.550000  1.700000e-20  1.035000e-16   95
#max    1522.000000  3.000000e-05  121.700000  8.400000e-06  7.000000e-02   120

# notice the std is huge for all these, filter the dataset on number of domains contained
enough_motifs = neg_matches.groupby('qname').filter(lambda x: len(x) > 2)  # narrow to 139

# then check they have one of each
not_all_three = []
for qname in set(enough_motifs['qname']):
    tmp_df = enough_motifs[enough_motifs['qname'] == qname]
    if len(set(tmp_df['acc'])) < 3:
        not_all_three.append(qname)

# drop em
not_in = enough_motifs[~enough_motifs['qname'].isin(not_all_three)]

endo_Cneg = not_in[not_in['acc'] == 'PF03944.13']
endo_Cneg.describe()

endo_Mneg = not_in[not_in['acc'] == 'PF00555.18']
endo_Mneg.describe()

endo_Nneg = not_in[not_in['acc'] == 'PF03945.13']
endo_Nneg.describe()




# against ours
# we build a custom HMM refrence from our three domains of interest obtained at: http://pfam.xfam.org/

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



# pca
qlen
must have all 3
evalue for each motifs_of_interest
score for each
bias for each
c-value, i-value



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