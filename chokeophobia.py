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
from pydpi.pypro import PyPro
from sklearn import model_selection
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import utils

# cwd
os.chdir('/Users/joannadreux/PycharmProjects/protein-classification/')

### Collect and clean data  ###
# Load the training set data
training_set = 'drugpro_training_set.csv'
ts = pd.read_csv(training_set, header=0)
ts.groupby('label').size()
# label
# 0    220
# 1     44

# Write FASTAs from training set data
subject = 'training_seqs_valid.fasta'
query = 'training_seqs_all.fasta'
valid_list = []
all_prots = []
for id, seq, label in ts.itertuples(index=False):
    seq_record = SeqRecord(Seq(seq, IUPAC.protein), id=id, description="Label:"+str(label), name="")
    all_prots.append(seq_record)
    if label == 1:
            valid_list.append(seq_record)
handle = open(subject, "w")
handle2 = open(query, "w")
for sequences in valid_list:
    SeqIO.write(sequences, handle, "fasta")
for sequences in all_prots:
    SeqIO.write(sequences, handle2, "fasta")


## Collect BLASTp data - sequence homology ##
out_blastp = 'blast/blastp_out.tab'
df_blast_master = utils.run_parse_blast(query, subject, 1, out_blastp)
# tidy the table
df_blast = df_blast_master[df_blast_master['pident'] != 100.0]  # remove identical hits
df_blast = df_blast.drop('sseqid', axis=1) # drop the reference column
df_blast = df_blast.drop_duplicates(subset=['qseqid'], keep='first') # only keep top hit, alignments reported by top hit
print df_blast.shape


## Collect domain similarity data - Hidden Markov Models ##
# first, run valid prots against Pfam-A.hmm to find domains of interest
out_hmmer = 'out_hmmer.txt'
pfam_db = '/Users/joannadreux/Desktop/hmm/Pfam-A.hmm'
df_hmm = utils.run_parse_hmmer(1e-5, out_hmmer, pfam_db, subject)
print df_hmm.shape
print len(set(df_hmm['qname'])) # all are represented
print set(df_hmm['acc'])  # only 3 domains
df_hmm.groupby('qname')['acc'].count()  # one missing, one has only one

# We run the analysis again but against each endo domain individually
# use .hmm files obtained from - http://pfam.xfam.org/
out_endoC = 'out_endoC.txt'
out_endoM = 'out_endoM.txt'
out_endoN = 'out_endoN.txt'
endoC = './endotoxins-hmm/Endotoxin_C.hmm'
endoM = './endotoxins-hmm/Endotoxin_M.hmm'
endoN = './endotoxins-hmm/Endotoxin_N.hmm'

# generate a df for each
df_C = utils.run_parse_hmmer(1, out_endoC, endoC, query)  # threshold = 1, allow all alignments
df_M = utils.run_parse_hmmer(1, out_endoM, endoM, query)
df_N = utils.run_parse_hmmer(1, out_endoN, endoN, query)

# tidy up the tables
df_C = utils.clean_up_hmm(df_C, 'C')
df_M = utils.clean_up_hmm(df_M, 'M')
df_N = utils.clean_up_hmm(df_N, 'N')

# Concat all 3 domain tables into one
assert set(df_C.qname_C == df_N.qname_N) == {True} # ensure all queries are represented and sorted
assert set(df_C.qname_C == df_M.qname_M) == {True}

# drop duplicated columns
df_M = df_M.drop(['qname_M', 'qlen_M'], axis=1)
df_N = df_N.drop(['qname_N', 'qlen_N'], axis=1)
df_endo = pd.concat([df_C, df_M, df_N], ignore_index=True, axis=1)
df_endo.columns = list(df_C.columns) + list(df_M.columns) + list(df_N.columns)
print df_endo.shape


## Collect physicochemical features homology -- pydpi ##
annotation_csv = './physicochem_annot.csv'
# write annotation table of features for each protein in ts
with open(annotation_csv, 'w') as f:
    for record in SeqIO.parse(query, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        desc = protein.GetCTD()
        moran = protein.GetMoranAuto()
        label = str(record.description).strip().split(':')[1]
        id = str(record.id)
        row = [id, label] + [str(i) for i in desc.values()] + [str(i) for i in moran.values()]
        f.write(','.join(row) + '\n')

df_desc = pd.read_table(annotation_csv, header=None, sep=',')
header = desc.keys() + moran.keys()
df_desc.columns = ['id', 'label'] + header


## Build the master df with all three sets of data ##
print df_desc.shape
print df_endo.shape
print df_blast.shape

# Remove missing values
prots_in_endo_df = list(df_endo['qname_C'])
df_descf = df_desc[df_desc['id'].isin(prots_in_endo_df)]
df_blastf = df_blast[df_blast['qseqid'].isin(prots_in_endo_df)]
print df_descf.shape
print df_endo.shape
print df_blastf.shape

# make protein names are sorted and matching
df_descf = df_descf.sort_values(by='id')
df_blastf= df_blastf.sort_values(by='qseqid')
df_endo = df_endo.sort_values(by='qname_C')
assert set(df_descf['id'] == df_blastf['qseqid']) == {True}
assert set(df_descf['id'] == df_endo['qname_C']) == {True}

# drop redundant cols and merge
df_endo = df_endo.drop(['qname_C'], axis=1)
df_blastf = df_blastf.drop(['qseqid'], axis=1)
dfs = [df_descf, df_blastf, df_endo]
for df in dfs:
    df.reset_index(drop=True, inplace= True)
master = pd.concat(dfs, axis=1, ignore_index=True)
master.columns = list(df_descf.columns) + list(df_blastf.columns) + list(df_endo.columns)
print master.shape
print master.columns


### Feature Selection ###
array = master.values
X = array[:,2:]
Y = list(array[:,1])  # labels
feature_names = list(master.columns[2:])
hits = utils.select_features(X, Y, 40, feature_names)
top_hits = list({k: v for k, v in hits.iteritems() if v > 1})  # use features flagged by two or more methods
print "Found {} selected features: {}".format(len(top_hits), top_hits)

# filter master
masterf = master[['id', 'label'] + top_hits]

# plot the data
matches = masterf[masterf['label'] == 1]  # valid prots
non_matches = masterf[masterf['label'] == 0]  # invalid prots
fig, axes = plt.subplots(6, 4)
for ax, val in zip(axes.flatten(), masterf.columns[2:]):
    ax.hist(matches[val].values, alpha=0.5)
    ax.hist(non_matches[val].values, alpha=0.5)
    ax.set_title(val)
fig.set_tight_layout(True)

# PCA
X = masterf.ix[:,2:].values
y = masterf['label']
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print 'Pct variance explained by top 2 components: {}'.format(str(pca.explained_variance_ratio_))
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)



### Model seletion ###

## Create Validation Dataset
labels = masterf['label']
features = masterf.ix[:, 2:].values
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=0.3)


## Try out multiple models
models = []
models.append(('LR', LogisticRegression()))  # linear
models.append(('LinearSVC', LinearSVC())) # linear
models.append(('KNN', KNeighborsClassifier()))  # non-linear
models.append(('NB', GaussianNB()))  # non-linear
models.append(('SVM', SVC()))  # non-linear

## Use 10 fold cross validation to estimate accuracy of each model tested
seed = 7  # using same random seed to make results comparable
scoring = 'accuracy'  # ratio correct predictions: total predictions * 100

# Compare Algorithms
results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    result = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(result)
    print '{}: {}, ({})'.format(name, result.mean(), result.std())

# plot results
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
plt.ylim([0.8, 1.001])
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(['LR', 'LinearSVC', "KNN", 'NB', 'SVM'])
plt.show()


### Make predictions
#The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the
# model on our validation set.
#This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation
# set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will
# result in an overly optimistic result.
#We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a
# confusion matrix and a classification report.

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)


svc = LogisticRegression()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)

# We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made.
# Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support
# showing excellent results (granted the validation dataset was small).
