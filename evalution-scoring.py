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
from chokeophobia import run_parse_blast, run_parse_hmmer, impute_values_blast, impute_values_hmm

query = 'sg_putative_10000.faa'
valid_fasta = 'training_seqs_valid.fasta'
all_fasta = 'training_seqs_all.fasta'
ts = pd.read_csv('drugpro_training_set.csv', header=0)
id_list = list(ts['id'])

def collect_data(query, valid_fasta, all_fasta, id_list):


    ## Collect BLASTp data - sequence homology ##
    out_blastp = 'blastp_out_targets.tab'
    df_blast_master = run_parse_blast(query, valid_fasta, 1, out_blastp)
    df_blast = df_blast_master.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit per query
    df_blast = impute_values_blast(id_list, df_blast)
    print df_blast.shape


    df_blast = df_blast_master[df_blast_master['pident'] != 100.0]  # remove identical hits
    df_blast = df_blast.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit per query
    df_blast = impute_values_blast(id_list, df_blast)
    print df_blast.shape

    ## Collect domain similarity data ##
    # generate a df for each
    df_C = run_parse_hmmer(1, 'out_endoC_targets.txt', './endotoxins-hmm/Endotoxin_C.hmm', all_fasta, 'C')
    df_M = run_parse_hmmer(1, 'out_endoM_targets.txt', './endotoxins-hmm/Endotoxin_M.hmm', all_fasta, 'M')
    df_N = run_parse_hmmer(1, 'out_endoN_targets.txt', './endotoxins-hmm/Endotoxin_N.hmm', all_fasta, 'N')

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
    dfs = [df_Cf, df_Mf, df_Nf]
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
    df_endo = pd.concat(dfs, ignore_index=True, axis=1)
    df_endo.columns = list(df_Cf.columns) + list(df_Mf.columns) + list(df_Nf.columns)
    print df_endo.shape


    ## Collect physicochemical features homology -- pydpi ##
    annotation_csv = './physicochem_annot_training.csv'
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

    return master


def score(df, predictions):
    """ Score results using the dictionary"""

    pkl_file = 'protein_nomination_value.pkl'
    money = pickle.load(open(pkl_file, "rb"))

    df['label'] = predictions
    match = list(df[df['label'] == 1]['id'])

    # add up your $$$
    total = 0
    for k, v in money.iteritems():
        if k in match:
            total += v
    print "You have made ${}!".format(total)

    return total

test_master = collect_data(query, valid_fasta, all_fasta, id_list)

# filter master on selected features
keep = tree_ft
masterf = test_master[['id'] + keep]


### Make predictions ###
test_array = masterf.values[:,1:]
# show time
predictions = knn.predict(test_array)
predictions2 = lin.predict(test_array)
predictions3 = nb.predict(test_array)

score(masterf, predictions)
score(masterf, predictions2)
score(masterf, predictions3)


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


