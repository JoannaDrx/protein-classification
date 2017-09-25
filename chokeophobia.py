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
from Bio.Blast.Applications import NcbiblastpCommandline
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# write fns for data collection
def collect_blast_data(all_fasta, valid_fasta, out):
    """" Runs the BLASTp job
        Parses the output
        Imputes missing values"""

    # run the BLASTp job
    blastp_cline = NcbiblastpCommandline(query=all_fasta, subject=valid_fasta, evalue=1, outfmt=6, out=out)
    stdout, stderr = blastp_cline()

    # read & filter the blast results
    df_master = pd.read_table(out, header=None)
    blast_header = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
    df_master.columns = blast_header
    df_master = df_master.drop(['sseqid', 'mm', 'gapopen', 'qstart', 'qend', 'sstart', 'send'], axis=1)

    # tidy the table
    df = df_master[df_master['pident'] != 100.0]  # remove identical hits
    df = df.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit per query

    # impute missing values, for queries where no alignment found
    # qseqid = id, pident = 0, len = 0, evalue = 1, bitscore =0
    tmp = pd.DataFrame([[seq, 0, 0, 1, 0] for seq in id_list if seq not in dft.values[:, 0]])
    tmp.columns = df.columns
    df = pd.concat([df, tmp])
    df = df.sort_values(by=df.columns[0])
    print df.shape

    return df

def collect_hmm_data(pfam_db, subject, suffix, id_list, out):
    """Runs a hmmscan job then parses the output into a pandas df"""

    cmd = 'hmmscan --domE 1 --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(out, pfam_db, subject)
    os.system(cmd)
    # read the results, select relevant cols
    df = pd.read_table(out, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                           usecols=[0,1,3,5,6,7,11,12,13,15,16,17,18,19,20,21])
    hmm_header = 'target acc qname qlen e-value score c-evalue i-evalue domain-score hf ht af at ef et reliability'.strip().split(' ')
    df.columns = hmm_header

    if suffix != '':
        df = df.drop(['target', 'acc', 'qlen', 'hf', 'ht', 'af', 'at', 'ef', 'et'], axis=1)
        df['count'] = df.groupby('qname')['qname'].transform('count')
        df = df.groupby('qname', as_index=False).mean()  # average out multiple domains
        df = df.add_suffix('_' + suffix)
        print df.shape

    # Impute missing values as follows, for queries where no domain was found:
    #    - qname = qname
    #    - e-value = 1
    #    - score = 0
    #    - c-evalue = 1
    #    - i-evalue = 1
    #    - domain_score = 0
    #    - reliability = 0
    #    - count = 0

    tmp = pd.DataFrame([[seq, 1,0,1,1,0,0,0] for seq in id_list if seq not in df.values[:,0]])
    tmp.columns = df.columns
    df = pd.concat([df, tmp])
    df = df.sort_values(by=df.columns[0])

    return df

def collect_pydpi_data():
    # write annotation table of features for each protein in ts
    with open(annotation_csv, 'w') as f:
        for record in SeqIO.parse(all_fasta, "fasta"):
            protein = PyPro()
            protein.ReadProteinSequence(str(record.seq))
            desc = protein.GetCTD()
            len_p = str(len(record.seq))
            label = str(record.description).strip().split(':')[1]
            id = str(record.id)
            row = [id, label, len_p] + [str(i) for i in desc.values()]
            f.write(','.join(row) + '\n')

    df_desc = pd.read_table(annotation_csv, header=None, sep=',')
    header = desc.keys()
    df_desc.columns = ['id', 'label', 'seq_length'] + header

    return df

# cwd
os.chdir('/Users/joannadreux/PycharmProjects/protein-classification/')

### Collect and clean data  ###
# Load the training set data
ts = pd.read_csv('drugpro_training_set.csv', header=0)
id_list = list(ts['id'])
ts.groupby('label').size()
# label
# 0    220
# 1     44

# Write FASTAs from training set data
valid_fasta = 'training_seqs_valid.fasta'
all_fasta = 'training_seqs_all.fasta'
valid_list = []
all_prots = []
for id, seq, label in ts.itertuples(index=False):
    seq_record = SeqRecord(Seq(seq, IUPAC.protein), id=id, description="Label:"+str(label), name="")
    all_prots.append(seq_record)
    if label == 1:
            valid_list.append(seq_record)
handle = open(valid_fasta, "w")
for sequences in valid_list:
    SeqIO.write(sequences, handle , "fasta")
handle2 = open(all_fasta, "w")
for sequences in all_prots:
    SeqIO.write(sequences, handle2, "fasta")


## Collect BLASTp data - sequence homology ##
out_blastp = 'blastp_out_training.tab'
df_blast =  collect_blast_data(out_blastp)

    df_blast_master = run_parse_blast(all_fasta, valid_fasta, 1, out_blastp)
    # tidy the table
    df_blast = df_blast_master[df_blast_master['pident'] != 100.0]  # remove identical hits
    df_blast = df_blast.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit per query
    df_blast = impute_values_blast(id_list, df_blast)
    print df_blast.shape


## Collect domain similarity data ##
# first, run valid prots against Pfam-A.hmm to find domains of interest
out_hmmer = 'out_hmmer.txt'
pfam_db = '/Users/joannadreux/Desktop/hmm/Pfam-A.hmm'
df_hmm = run_parse_hmmer(1e-5, out_hmmer, pfam_db, valid_fasta, '')
print df_hmm.shape
print len(set(df_hmm['qname'])) # all valid proteins are represented
print set(df_hmm['acc'])  # only 3 domains are aligned to by the valid proteins
print set(df_hmm.groupby('qname')['acc'].count())  # and they all have all 3

# We run the analysis again but against each domain of interest individually
# use .hmm files obtained from - http://pfam.xfam.org/
df_C = run_parse_hmmer(1, 'out_endoC_training.txt', './endotoxins-hmm/Endotoxin_C.hmm', all_fasta, 'C')
df_M = run_parse_hmmer(1, 'out_endoM_training.txt', './endotoxins-hmm/Endotoxin_M.hmm', all_fasta, 'M')
df_N = run_parse_hmmer(1, 'out_endoN_training.txt', './endotoxins-hmm/Endotoxin_N.hmm', all_fasta, 'N')

# impute missing values
df_Cf = impute_values_hmm(id_list, df_C)
df_Mf = impute_values_hmm(id_list, df_M)
df_Nf = impute_values_hmm(id_list, df_N)

# Concat all 3 domain tables into one
assert df_Cf.shape == df_Mf.shape == df_Nf.shape
assert set(df_Cf.qname_C == df_Nf.qname_N) == {True}
assert set(df_Cf.qname_C == df_Mf.qname_M) == {True}
df_Mf = df_Mf.drop(['qname_M'], axis=1)  # drop duplicated columns
df_Nf = df_Nf.drop(['qname_N'], axis=1)
dfs=[df_Cf, df_Mf, df_Nf]
for df in dfs:
    df.reset_index(drop=True, inplace=True)
df_endo = pd.concat(dfs, ignore_index=True, axis=1)
df_endo.columns = list(df_Cf.columns) + list(df_Mf.columns) + list(df_Nf.columns)
print df_endo.shape


## Collect physicochemical features homology -- pydpi ##
annotation_csv = './physicochem_annot_training.csv'




## Build the master df with all three sets of data ##
print df_desc.shape
print df_endo.shape
print df_blast.shape

# make protein names are sorted and matching
df_desc = df_desc.sort_values(by='id')
df_blast= df_blast.sort_values(by='qseqid')
df_endo = df_endo.sort_values(by='qname_C')
assert set(df_desc['id'] == df_blast['qseqid']) == {True}
assert set(df_desc['id'] == df_endo['qname_C']) == {True}

# drop redundant cols and merge
df_endo = df_endo.drop(['qname_C'], axis=1)
df_blast = df_blast.drop(['qseqid'], axis=1)
dfs = [df_desc, df_blast, df_endo]
for df in dfs:
    df.reset_index(drop=True, inplace= True)
master = pd.concat(dfs, axis=1, ignore_index=True)
master.columns = list(df_desc.columns) + list(df_blast.columns) + list(df_endo.columns)
print master.shape
print master.columns


### Feature Selection ###
array = master.values
X = array[:,2:]
Y = list(array[:,1])  # labels
feature_names = list(master.columns[2:])

# K Best
test = SelectKBest(k=30)
fit = test.fit(X, Y)
l = []
for bool, ft in zip(fit.get_support(), feature_names):
    if bool:
        l.append(ft)
print 'Select K Best selected features:', l

# Feature Extraction with RFE
model = LogisticRegression()
rfe = RFE(model, 30)
fit = rfe.fit(X, Y)
k = []
for bool, ft in zip(fit.support_, feature_names):
    if bool:
        k.append(ft)
print 'RFE selected features:', k

# Feature Importance with Extra Trees Classifier
model = ExtraTreesClassifier()
model.fit(X, Y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
tree_ft = []
for i in range(30):
    tree_ft.append(feature_names[indices[i]])
print 'Extra Trees Classifier selected features:', tree_ft


# filter master
mk = master[['id', 'label'] + l]
ml = master[['id', 'label'] + k]
mt = master[['id', 'label'] + tree_ft]

sel_df = mt
### Model selection ###

## Create Validation Dataset
labels = sel_df['label']
features = sel_df.ix[:, 2:].values
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

### Make predictions
# knn
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)

# linear regression
lin = LogisticRegression()
lin.fit(X_train, Y_train)
predictions2 = lin.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)

# gaussian NB
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions3 = nb.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)


### PLOTS
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


# plot results
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
plt.ylim([0.8, 1.001])
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(['LR', 'LinearSVC', "KNN", 'NB', 'SVM'])
plt.show()