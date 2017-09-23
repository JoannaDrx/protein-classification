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
#from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pydpi.pypro import PyPro
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# cwd
os.chdir('/Users/joannadreux/PycharmProjects/protein-classification/')

### Load the Data
training_set = 'drugpro_training_set.csv'
ts = pd.read_csv(training_set, header=0)
ts.groupby('label').size()
# label
# 0    220
# 1     44

### BLAST - sequence homology
subject = 'training_seqs_valid.fasta'
query = 'training_seqs_all.fasta'
out_blastp = 'blast/blastp_out.tab'

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

def run_parse_blast(query, subject, threshold, out_file):

    # run blastp
    blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=threshold, outfmt=6, out=out_file)
    stdout, stderr = blastp_cline()

    # read & filter the blast results
    df = pd.read_table(out_blastp, header=None)
    blast_header = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
    df.columns = blast_header
    return df

df_blast = run_parse_blast(query, subject, 1, out_blastp)
# filter
df_blast = df_blast[df_blast['pident'] != 100.0]  # remove identical hits
print df_blast.shape
# only keep top hit, alignments are reported by best hit
df_blast = df_blast.drop_duplicates(subset=['qseqid'], keep='first')
print df_blast.shape

# clean up BLAST data - drop the reference column and add a Label column to see correlation
df_blast = df_blast.drop('sseqid', axis=1)
df_blast['label'] = 0  # set default value
valids = ts[ts['label'] == 1]['id']  # list of valid prots ids
df_blast.loc[(df_blast['qseqid'].isin(valids)), 'label'] = 1  # set to 1 for valids

# find correlation (+/- 0.5)
print df_blast.corr()['label']
co_cols = df_blast.corr()['label']
keep = co_cols[(co_cols > 0.7) | (co_cols < -0.7)]  # keep % identical bases, mismatches & bitScore
df_blast = df_blast[['qseqid'] + list(keep.index)]

# plot the two distributions
matches = df_blast[df_blast['label'] == 1]  # valid prot alignments, all 44
non_matches = df_blast[df_blast['label'] == 0]  # invalid prot alignments
fig, axes = plt.subplots(3, 1)
for ax, val in zip(axes.flatten(), df_blast.columns[1:4]):
    ax.hist(matches[val].values, alpha=0.5)
    ax.hist(non_matches[val].values, alpha=0.5)
    ax.set_title(val)
fig.set_tight_layout(True)


### Hidden Markov Models -- domain similarity
# against valid proteins in training set, find domains of interest
out_hmmer = 'out_hmmer.txt'
pfam_db = '/Users/joannadreux/Desktop/hmm/Pfam-A.hmm'

def run_parse_hmmer(threshold, out_file, pfam_db, subject):

    cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_file, pfam_db,
                                                                                      subject)
    os.system(cmd)
    # read the results, select relevant cols
    df = pd.read_table(out_file, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                           usecols=[0,1,3,5,6,7,11,12,13,15,16,17,18,19,20,21])
    hmm_header = 'target acc qname qlen e-value score c-evalue i-evalue domain-score hf ht af at ef et reliability'.strip().split(' ')
    df.columns = hmm_header
    return df

# run analysis on label=1 only
df_hmm = run_parse_hmmer(1e-5, out_hmmer, pfam_db, subject)
print df_hmm.shape
print len(set(df_hmm['qname']))
print set(df_hmm['acc'])  # only 3 domains


# we build a custom HMM reference from our three domains of interest - http://pfam.xfam.org/
# run the analysis again against this reference with the whole training set
out_hmmer_endo = 'out_hmmer_endo_only.txt'
pfam_db_endo = './endotoxins-hmm/endotoxins.hmm'
df_endo = run_parse_hmmer(1, out_hmmer_endo, pfam_db_endo, query)  # threshold = 1, allow all alignments

# we need one row per qname
df_endo = df_endo.groupby('qname', as_index=False).sum()

# add label
df_endo['label'] = 0  # set default value
df_endo.loc[(df_endo['qname'].isin(valids)), 'label'] = 1  # set to 1 for valids


### Physicochemical features homology using pydpi
annotation_csv = './physicochem_annot.csv'
# go through fasta and look at all pydpi sats, write annotation table
with open(annotation_csv, 'w') as f:
    for record in SeqIO.parse(query, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        desc = protein.GetCTD()
        label = str(record.description).strip().split(':')[1]
        id = str(record.id)
        f.write(id + ',' + ','.join([str(i) for i in desc.values()]) + ',' + label + '\n')

desc_header = desc.keys()
df_desc = pd.read_table(annotation_csv, header=0, sep=',')
df_desc.columns = ['id'] + desc_header + ['label']


## Build master df to select features on
print ts.shape
print df_desc.shape
print df_endo.shape
print df_blast.shape

df_blast_full = df_blast.sort_values(by='qseqid')
df_ctd_full = df_desc.sort_values(by='id')
df_hmm_full = df_endo.sort_values(by='qname')
ts = ts.sort_values(by='id')

t = ts.drop(['seq'], axis=1)
b = df_blast_full.drop(['qseqid', 'label'], axis=1)
c = df_ctd_full.drop(['label', 'id'], axis=1)
h = df_hmm_full.drop(['qname', 'label'], axis=1)

frames = [t, b, c, h]
master = pd.concat(frames, ignore_index=True, axis=1)
master.columns = list(t.columns) + list(b.columns) + list(c.columns) + list(h.columns)
print master.shape




# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
df_desc.shape
array = df_desc.values
X = array[:,1:148]
Y = list(array[:,148])

# feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

feature_names = list(df_desc.columns[1:148])
mask = test.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
print new_features


# Feature Extraction with RFE
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

# Feature Importance with Extra Trees Classifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)






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


