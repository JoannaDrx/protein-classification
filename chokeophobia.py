"""
Second Genome - Genomic Data Scientist challenge
Author: Joanna Dreux [joanna.dreux@gmail.com]
"""

import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.Blast.Applications import NcbiblastpCommandline
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pydpi.pypro import PyPro
import pickle

# cwd
os.chdir('/Users/joanna/PycharmProjects/protein-classification/')

### Load the Data
training_set = 'drugpro_training_set.csv'
names = ['id', 'seq', 'label']
ts = pd.read_csv(training_set, header=0)
print ts.shape
ts.groupby('label').size()
# label
# 0    220
# 1     44

### BLAST - sequence homology
subject = 'training_seqs_valid.fasta'
query = 'training_seqs_all.fasta'
out_blastp = 'BLAST/blastp_out.tab'

# Convert training set to FASTA file format, annotate with label
valid_list = []
all = []
for id, seq, label in ts.itertuples(index=False):
    seq_record = SeqRecord(Seq(seq, IUPAC.protein), id=id, description="Label:"+str(label), name="")
    all.append(seq_record)
    if label == 1:
            valid_list.append(seq_record)
handle = open(subject, "w")
handle2 = open(query, "w")
for sequences in valid_list:
    SeqIO.write(sequences, handle, "fasta")
for sequences in all:
    SeqIO.write(sequences, handle2, "fasta")

# blastp against the training set
blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=1, outfmt=6, out=out_blastp)
stdout, stderr = blastp_cline()

# read & filter the blast results
df = pd.read_table(out_blastp, header=None)
blast_header = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
df.columns = blast_header

# filter
df_blast = df[df['pident'] != 100.0] # remove identical hits
# only keep top hit (we report max_target_seqs=2 for self-aligners)
df_blast = df_blast.sort_values('evalue').drop_duplicates(subset=['qseqid'], keep='first')

### Hidden Markov Models -- domain similarity
# against training set, find domains of interest
out_hmmer = 'out_hmmer.txt'
pfam_db = '/Users/joanna/Desktop/hmm-db/Pfam-A.hmm'
threshold = 1e-5
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, query)
os.system(cmd)

# read the results, select relevant cols
df_hmm_all = pd.read_table(out_hmmer, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                       usecols=[0,1,3,5,6,7,11,12,13,15,16,17,18,19,20,21])
hmm_header = 'target acc qname qlen e-value score cond-e-value ind-e-value domain-score hf ht af at ef et reliability'.strip().split(' ')
df_hmm_all.columns = hmm_header

# valid proteins
pos_matches = df_hmm_all[df_hmm_all['qname'].str.contains('label:1')]
len(set(pos_matches['qname']))  # check they're all represented
motifs_of_interest = set(pos_matches['acc'])  # only 3 domains

# negative proteins
neg_matches = df_hmm_all[df_hmm_all['qname'].str.contains('label:0')]
neg_matches = neg_matches[neg_matches['acc'].str.contains('PF03945.13|PF00555.18|PF03944.13')] # limit to these

# does each prot have all 3 domain?
def check_domains(df):
    for qname in set(df['qname']):
        tmp_df = df[df['qname'] == qname]
        if len(set(tmp_df['acc'])) < 3:
            print 'missing domain in {}'.format(qname)
    return

check_domains(pos_matches)  # all valid prots have all 3
check_domains(neg_matches)  # varies

# we build a custom HMM refrence from our three domains of interest obtained at: http://pfam.xfam.org/
# run the analysis again against this refernce
out_hmmer = 'out_hmmer_endo_only.txt'
pfam_db = './endotoxins-hmm/endotoxins.hmm'
threshold = 1e-5
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, query)
os.system(cmd)

# read the results, select relevant cols
df_hmm = pd.read_table(out_hmmer, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                       usecols=[0,1,3,5,6,7,11,12,13,15,16,17,18,19,20,21])
hmm_header = 'target acc qname qlen e-value score cond-e-value ind-e-value domain-score hf ht af at ef et reliability'.strip().split(' ')
df_hmm.columns = hmm_header


### Physicochemical features homology using pydpi
annotation_csv = './pydpi/physicochem_annot.csv'
# go through fasta and look at a CDT sats, write annotation table
with open(annotation_csv, 'w') as f:
    for record in SeqIO.parse(query, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        ctd = protein.GetCTD()
        label= str(record.description).strip().split(':')[1]
        id = str(record.id)
        f.write(id + ',' + label + ',' + ','.join([str(i) for i in ctd.values()]) + '\n')
ctd_header = ctd.keys()
df_ctd = pd.read_table(annotation_csv, header=0, sep=',')
df_ctd.columns = ['id', 'Label'] + ctd_header


### Clean-up the data, keep only highly correlated variables

# clean up BLAST data
print df_blast.shape
print df_blast.columns

# drop the reference column and add a Label column to see correlation
df_blast = df_blast.drop('sseqid', axis=1)
valids = ts[ts['label'] == 1]['id']  # list of valid prots ids
df_blast['label'] = 0  # set default value
df_blast['label'][df_blast['qseqid'].isin(valids)] = 1  # set to 1 by cross referencing with original ts

# find correlation (+/- 0.5)
print df_blast.corr()['label']

# keep correlated columns, drop the rest
df_blast = df_blast[['qseqid', 'pident', 'len', 'label']]
matches = df_blast[df_blast['label'] == 1]  # valid prot alignments
non_matches = df_blast[df_blast['label'] == 0]  # invalid prot alignments
# plot some
fig, axes = plt.subplots(4, 1)
for ax, val in zip(axes.flatten(), df_blast.columns[1:]):
    ax.hist(matches[val].values, alpha=0.5)
    ax.hist(non_matches[val].values, alpha=0.5)
    ax.set_title(val)
fig.set_tight_layout(True)

# fill in the blanks
training_ids = ts['id']
data = pd.DataFrame([])
for missing in training_ids[~training_ids.isin(df_blast['qseqid'])]:
    data = data.append(pd.DataFrame({'qseqid':missing, 'pident':0, 'len':0,'label':ts[ts['id'] == missing]['label']}),
                       ignore_index=True)

frames = [df_blast, data]
df_blast_full = pd.concat(frames, ignore_index=True)
print df_blast_full.shape


## clean up HMM data
print df_hmm.shape # missing when no domain, add extra columns for each of the three domains
print df_hmm.columns

# add a Label column to see correlation
df_hmm['label'] = 0  # set default value
df_hmm['label'][df_hmm['qname'].isin(valids)] = 1  # set to 1 by cross referencing with original ts

# first look at correlation
print df_hmm.corr()['label']  # keep score, domain-score, reliability
df_hmm = df_hmm.drop(['target', 'qlen', 'e-value', 'cond-e-value', 'ind-e-value','hf', 'ht', 'af', 'at', 'ef', 'et'],
                     axis=1)
print df_hmm.shape

# join rows of same query into single row
df_hmm['endoM-score'] = 0  # set default value
df_hmm['endoM-score'][df_hmm['acc'] == 'PF00555.18'] = df_hmm['score']

df_hmm['endoM-dscore'] = 0  # set default value
df_hmm['endoM-dscore'][df_hmm['acc'] == 'PF00555.18'] = df_hmm['domain-score']

df_hmm['endoM-reliability'] = 0  # set default value
df_hmm['endoM-reliability'][df_hmm['acc'] == 'PF00555.18'] = df_hmm['reliability']

df_hmm['endoC-score'] = 0  # set default value
df_hmm['endoC-score'][df_hmm['acc'] == 'PF03944.13'] = df_hmm['score']

df_hmm['endoC-dscore'] = 0  # set default value
df_hmm['endoC-dscore'][df_hmm['acc'] == 'PF03944.13'] = df_hmm['domain-score']

df_hmm['endoC-reliability'] = 0  # set default value
df_hmm['endoC-reliability'][df_hmm['acc'] == 'PF03944.13'] = df_hmm['reliability']

df_hmm['endoN-score'] = 0  # set default value
df_hmm['endoN-score'][df_hmm['acc'] == 'PF03945.13'] = df_hmm['score']

df_hmm['endoN-dscore'] = 0  # set default value
df_hmm['endoN-dscore'][df_hmm['acc'] == 'PF03945.13'] = df_hmm['domain-score']

df_hmm['endoN-reliability'] = 0  # set default value
df_hmm['endoN-reliability'][df_hmm['acc'] == 'PF03945.13'] = df_hmm['reliability']

# drop duplicated columns
df_hmm = df_hmm.drop(['acc', 'score', 'domain-score', 'reliability'],axis=1)

# concatenate rows
df_hmm = df_hmm.groupby(['qname'], as_index=False).sum()
df_hmm['label'][df_hmm['label'] == 3] = 1  #woops
print df_hmm.corr()['label']

# fill in the blanks, missing ids
training_ids = ts['id']
data = pd.DataFrame([])
for missing in training_ids[~training_ids.isin(df_hmm['qname'])]:
    data = data.append(pd.DataFrame({'qname':missing, 'label':ts[ts['id'] == missing]['label'], 'endoM-score':0,
                                     'endoM-dscore':0,'endoM-reliability':0, 'endoC-score':0, 'endoC-dscore':0,
                                    'endoC-reliability':0, 'endoN-score':0, 'endoN-dscore':0,
                                    'endoN-reliability':0}), ignore_index=True)

frames = [df_hmm, data]
df_hmm_full = pd.concat(frames, ignore_index=True)
print df_hmm_full.shape

# Physicochemical properties clean-up
print df_ctd.shape  # many columns, select relevant ones
print df_ctd.columns

corr_cols = df_ctd.corr()['Label']
corr_cols.describe()
keep = corr_cols[(corr_cols > 0.5) | (corr_cols < -0.5)]
df_ctd = df_ctd[['id'] + list(keep.index)]
print df_ctd.shape

# fill in the blanks, missing ids
training_ids = ts['id']
data = pd.DataFrame([])
for missing in training_ids[~training_ids.isin(df_ctd['id'])]:
    data = data.append(pd.DataFrame({'id': missing,'label':ts[ts['id'] == missing]['label'],
                                     '_HydrophobicityT23': 0, '_ChargeD2075': 0,
                                    '_ChargeD3075': 0, '_SecondaryStrD2050': 0, '_SecondaryStrD2025': 0,
                                    '_PolarizabilityD3075': 0, '_PolarizabilityT12': 0, '_SecondaryStrD2001': 0,
                                     '_SecondaryStrT13': 0, '_NormalizedVDWVD3075': 0, '_PolarizabilityD2075': 0}),
                       ignore_index=True)

frames = [df_ctd, data]
df_ctd_full = pd.concat(frames, ignore_index=True)
print df_ctd_full.shape

## Build master df to train model on
print ts.shape
print df_ctd_full.shape
print df_hmm_full.shape
print df_blast_full.shape

df_blast_full = df_blast_full.sort_values(by='qseqid')
df_ctd_full = df_ctd_full.sort_values(by='id')
df_hmm_full = df_hmm_full.sort_values(by='qname')
ts = ts.sort_values(by='id')

t = ts.drop(['seq'], axis=1)
b = df_blast_full.drop(['qseqid', 'label'], axis=1)
c = df_ctd_full.drop(['label', 'id'], axis=1)
h = df_hmm_full.drop(['qname', 'label'], axis=1)

frames = [t, b, c, h]
master = pd.concat(frames, ignore_index=True, axis=1)
master.columns = list(t.columns) + list(b.columns) + list(c.columns) + list(h.columns)
print master.shape


## Make training/validation datasets
label_names = ['0', '1']
labels = master['label']
feature_names = master.columns[2:]
features = master.ix[:,2:].values

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=validation_size,
                                                                                random_state=seed)
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))

# eval
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


