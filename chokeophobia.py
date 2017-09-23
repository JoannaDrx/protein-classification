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
from collections import Counter
from operator import itemgetter
from pandas.tools.plotting import scatter_matrix
#from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
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
import utils

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

# run BLASTp
df_blast = utils.run_parse_blast(query, subject, 1, out_blastp)
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

# run analysis on label=1 only
df_hmm = utils.run_parse_hmmer(1e-5, out_hmmer, pfam_db, subject)
print df_hmm.shape
print len(set(df_hmm['qname']))
print set(df_hmm['acc'])  # only 3 domains
df_hmm.groupby('qname')['acc'].count()  # one missing, one has only one

# We run the analysis again but against each endo domain individually
# use .hmm files obtained from - http://pfam.xfam.org/
out_endoC = 'out_endoC.txt'
out_endoM = 'out_endoM.txt'
out_endoN = 'out_endoN.txt'
endoC = './endotoxins-hmm/Endotoxin_C.hmm'
endoM = './endotoxins-hmm/Endotoxin_C.hmm'
endoN = './endotoxins-hmm/Endotoxin_C.hmm'

# generate a df for each
df_C = utils.run_parse_hmmer(1, out_endoC, endoC, query)  # threshold = 1, allow all alignments
df_M = utils.run_parse_hmmer(1, out_endoM, endoM, query)
df_N = utils.run_parse_hmmer(1, out_endoN, endoN, query)

# desc here
df_C = utils.clean_up_hmm(df_C,'C')
df_M = utils.clean_up_hmm(df_M, "M")
df_N = utils.clean_up_hmm(df_N, "N")

# Concat into one table
# first make sure all the queries are represented and sorted
assert set(df_C.qname_C == df_N.qname_N) == {True}
assert set(df_C.qname_C == df_M.qname_M) == {True}

# drop duplicated columns
df_M = df_M.drop(['qname_M', 'qlen_M'], axis=1)
df_N = df_N.drop(['qname_N', 'qlen_N'], axis=1)
frames = [df_C, df_M, df_N]
master = pd.concat(frames, ignore_index=True, axis=1)
master.columns = list(df_C.columns) + list(df_M.columns) + list(df_N.columns)
print master.shape

# add label column
master['label'] = 0  # set default value
master.loc[(master['qname_C'].isin(valids)), 'label'] = 1  # set to 1 for valids
print master.shape

# select features
array = master.values
X = array[:, 1:41]
Y = list(array[:, 41])
feature_names = list(master.columns[1:42])
hits = utils.select_features(X, Y, 10, feature_names)
# Select K Best selected features: ['qlen_C', 'domain-score_C', 'domain-score_M', 'hf_M', 'domain-score_N', 'hf_N']
# RFE selected features: ['af_C', 'ef_C', 'af_M', 'ef_M', 'af_N', 'ef_N']
# Extra Trees Classifier selected features: ['reliability_M', 'reliability_C', 'one_of_each', 'domain-score_M',
# 'score_N', 'hf_M']

# we retain score_N, score_M, score_C, hf_N, hf_M, hf_C
keep = ['score_N', 'score_M', 'score_C', 'hf_N', 'hf_M', 'hf_C']
master = master[['qname_C'] + keep]


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


