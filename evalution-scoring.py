"""
Second Genome - Genomic Data Scientist challenge
Author: Joanna Dreux [joanna.dreux@gmail.com]
"""

import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pydpi.pypro import PyPro
import pickle
import utils

query = 'sg_putative_10000.faa'

## Collect BLASTp data - sequence homology ##
out_blastp = 'blast/blastp_out_sgtargets.tab'
df_blast_master = utils.run_parse_blast(query, 'training_seqs_valid.fasta', 1, out_blastp)
df_blast = df_blast_master.drop('sseqid', axis=1) # drop the reference column
df_blast = df_blast.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit, alignments reported by top hit

## Collect domain similarity data - Hidden Markov Models ##
# generate a df for each
df_C = utils.run_parse_hmmer(1, 'out_endoC_targets.txt', './endotoxins-hmm/Endotoxin_C.hmm', query)
df_M = utils.run_parse_hmmer(1, 'out_endoM_targets.txt', './endotoxins-hmm/Endotoxin_M.hmm', query)
df_N = utils.run_parse_hmmer(1, 'out_endoN_targets.txt', './endotoxins-hmm/Endotoxin_N.hmm', query)

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
annotation_csv = './physicochem_annot_targets.csv'
# write annotation table of features for each protein in ts
with open(annotation_csv, 'w') as f:
    for record in SeqIO.parse(query, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        desc = protein.GetCTD()
        moran = protein.GetMoranAuto()
        id = str(record.id)
        row = [id] + [str(i) for i in desc.values()] + [str(i) for i in moran.values()]
        f.write(','.join(row) + '\n')

df_desc = pd.read_table(annotation_csv, header=None, sep=',')
header = desc.keys() + moran.keys()
df_desc.columns = ['id'] + header


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
# filter master
masterf = master[['id'] + top_hits]


### Make predictions ###
test_array = masterf.values[:,1:]
# show time
predictions = knn.predict(test_array)
predictions2 = svc.predict(test_array)



# PCA
X = masterf.ix[:,2:].values
y = predictions
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print 'Pct variance explained by top 2 components: {}'.format(str(pca.explained_variance_ratio_))
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)


### Score using the pickle file ###
pkl_file = 'protein_nomination_value.pkl'
money = pickle.load(open(pkl_file,"rb"))
print money
result_df = masterf
result_df['label'] = predictions

match = list(result_df[result_df['label']==1]['id'])

# add up your $$$
total = 0
for k, v in money.iteritems():
    if k in match:
        total += v

print "You have made ${}".format(total)