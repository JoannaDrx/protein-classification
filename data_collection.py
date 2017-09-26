"""Data Collection fns"""

import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from pydpi.pypro import PyPro
from Bio.Blast.Applications import NcbiblastpCommandline


# cwd
os.chdir('/Users/joannadreux/PycharmProjects/protein-classification/')

### Collect data from the training set  ###
# Load the training set data
ts = pd.read_csv('drugpro_training_set.csv', header=0)
id_list = list(ts['id'])
print ts.groupby('label').size()
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
    SeqIO.write(sequences, handle, "fasta")
handle2 = open(all_fasta, "w")
for sequences in all_prots:
    SeqIO.write(sequences, handle2, "fasta")


# write fns for data collection
def collect_blast_data(query, subject, id_list, out):
    """" Runs the BLASTp job
        Parses the output
        Imputes missing values"""

    # run the BLASTp job
    print 'Running BLASTp job...'
    blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=1e-3, outfmt=6, out=out)
    stdout, stderr = blastp_cline()

    # read & filter the blast results
    df_master = pd.read_table(out, header=None)
    blast_header = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
    df_master.columns = blast_header

    # tidy the table
    df = df_master[df_master['pident'] != 100.0]  # remove identical hits
    df = df.drop_duplicates(subset=['qseqid'], keep='first')  # only keep top hit per query
    df = df.drop(['sseqid', 'mm', 'gapopen', 'qstart', 'qend', 'sstart', 'send'], axis=1)

    print 'Imputing missing data...'
    # impute missing values, for queries where no alignment found
    # qseqid = id, pident = 0, len = 0, mm = evalue = 1, bitscore =0
    tmp = pd.DataFrame([[seq, 0, 0, 1, 0] for seq in id_list if seq not in df.values[:, 0]])
    tmp.columns = df.columns
    df = pd.concat([df, tmp])
    df = df.sort_values(by=df.columns[0])
    print 'Collected BLAST df: {}'.format(df.shape)

    return df


def collect_hmm_data(pfam_db, threshold, subject, suffix, id_list, out):
    """Runs a hmmscan job then parses the output into a pandas df"""

    print 'Running HMMscan job...'
    cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out, pfam_db, subject)
    os.system(cmd)
    # read the results, select relevant cols
    df = pd.read_table(out, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                           usecols=[0,1,3,5,6,7,11,12,13,15,16,17,18,19,20,21])
    hmm_header = 'target acc qname qlen e-value score c-evalue i-evalue domain-score hf ht af at ef et reliability'.strip().split(' ')
    df.columns = hmm_header

    print 'Found HMM df with : {}'.format(df.shape)

    if suffix != '':
        df = df.drop(['target', 'acc', 'qlen', 'hf', 'ht', 'af', 'at', 'ef', 'et'], axis=1)
        df['count'] = df.groupby('qname')['qname'].transform('count')
        df = df.groupby('qname', as_index=False).mean()  # average out multiple domains
        df = df.add_suffix('_' + suffix)

        # Impute missing values as follows, for queries where no domain was found:
        #    - qname = qname
        #    - e-value = 1
        #    - score = 0
        #    - c-evalue = 1
        #    - i-evalue = 1
        #    - domain_score = 0
        #    - reliability = 0
        #    - count = 0
        print 'Imputing missing data...'
        tmp = pd.DataFrame([[seq, 1,0,1,1,0,0,0] for seq in id_list if seq not in df.values[:,0]])
        tmp.columns = df.columns
        df = pd.concat([df, tmp])
        df = df.sort_values(by=df.columns[0])
        print 'Done.'

    return df


# Collect BLASTp data - sequence homology #
df_blast =  collect_blast_data(all_fasta, valid_fasta, id_list, 'blastp_out_training.tab')
print 'BLASTp table: {}'.format(df_blast.columns)

# Collected BLAST df: (264, 5)
# BLASTp table: Index([u'qseqid', u'pident', u'len', u'evalue', u'bitscore'], dtype='object')


# Collect domain similarity data #
# first, run valid prots against Pfam-A.hmm to find domains of interest
df_hmm = collect_hmm_data('/Users/jdreux/Desktop/hmm/Pfam-A.hmm', 1e-5, valid_fasta, '', id_list, 'hmm/out_hmmer.txt')

print len(set(df_hmm['qname']))  # all valid proteins are represented
# 44

print set(df_hmm['acc'])  # only 3 domains are aligned to by the valid proteins
# set(['PF03945.13', 'PF00555.18', 'PF03944.13'])

print set(df_hmm.groupby('qname')['acc'].count())  # and they all have all 3
# set([3])


# We run the analysis again but against each domain of interest individually
# use .hmm files obtained from - http://pfam.xfam.org/
df_C = collect_hmm_data('./endotoxins-hmm/Endotoxin_C.hmm', 1e-3, all_fasta, 'C', id_list, 'hmm/out_endoC_training.txt')
df_M = collect_hmm_data('./endotoxins-hmm/Endotoxin_M.hmm', 1e-3, all_fasta, 'M', id_list, 'hmm/out_endoM_training.txt')
df_N = collect_hmm_data('./endotoxins-hmm/Endotoxin_N.hmm', 1e-3, all_fasta, 'N', id_list, 'hmm/out_endoN_training.txt')

# Concat all 3 domain tables into one
assert df_C.shape == df_M.shape == df_N.shape
assert set(df_C.qname_C == df_N.qname_N) == {True}
assert set(df_C.qname_C == df_M.qname_M) == {True}
df_Mf = df_M.drop(['qname_M'], axis=1)  # drop duplicated columns
df_Nf = df_N.drop(['qname_N'], axis=1)
dfs=[df_C, df_Mf, df_Nf]
for df in dfs:
    df.reset_index(drop=True, inplace=True)
df_endo = pd.concat(dfs, ignore_index=True, axis=1)
df_endo.columns = list(df_C.columns) + list(df_Mf.columns) + list(df_Nf.columns)

print 'Endotoxin domains table ({}): {}'.format(df_endo.shape, df_endo.columns)
# Endotoxin domains table ((264, 22)): Index([u'qname_C', u'e-value_C', u'score_C', u'c-evalue_C', u'i-evalue_C',
       # u'domain-score_C', u'reliability_C', u'count_C', u'e-value_M',
       # u'score_M', u'c-evalue_M', u'i-evalue_M', u'domain-score_M',
       # u'reliability_M', u'count_M', u'e-value_N', u'score_N', u'c-evalue_N',
       # u'i-evalue_N', u'domain-score_N', u'reliability_N', u'count_N'],
       # dtype='object')


# Collect physicochemical features homology -- pydpi ##
print 'Collecting physicochemical stats per protein...'
with open('./physicochem_annot_training.csv', 'w') as f:
    for record in SeqIO.parse(all_fasta, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        desc = protein.GetMoreauBrotoAuto()
        desc2 = protein.GetAAComp()
        z = desc.copy()
        z.update(desc2)
        len_p = str(len(record.seq))
        label = str(record.description).strip().split(':')[1]
        id = str(record.id)
        row = [id, label, len_p] + [str(i) for i in z.values()]
        f.write(','.join(row) + '\n')

df_desc = pd.read_table('./physicochem_annot_training.csv', header=None, sep=',')
header = z.keys()
df_desc.columns = ['id', 'label', 'seq_length'] + header
print 'Pydpi table: {}'.format(df_desc.columns)


# Build the master df with all three sets of data ##
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
print 'Built master df ({}): {}'.format(master.shape, master.columns)

master.to_pickle('training_set.pkl')


### Collect data for the putative_10000 fasta ###
query = 'sg_putative_10000.faa'
id_list_test = []
fasta_sequences = SeqIO.parse(open(query),'fasta')
for fasta in fasta_sequences:
    id_list_test.append(str(fasta.id))

# Collect BLASTp data
df_bl =  collect_blast_data(query, valid_fasta, id_list_test, 'blastp_out_test.tab')
print 'BLASTp table: {}'.format(df_bl.columns)

# Collect domain similarity data
df_C = collect_hmm_data('./endotoxins-hmm/Endotoxin_C.hmm', 1e-3, query, 'C', id_list_test, 'hmm/out_endoC_training.txt')
df_M = collect_hmm_data('./endotoxins-hmm/Endotoxin_M.hmm', 1e-3, query, 'M', id_list_test, 'hmm/out_endoM_training.txt')
df_N = collect_hmm_data('./endotoxins-hmm/Endotoxin_N.hmm', 1e-3, query, 'N', id_list_test, 'hmm/out_endoN_training.txt')

# Concat all 3 domain tables into one
assert df_C.shape == df_M.shape == df_N.shape
assert set(df_C.qname_C == df_N.qname_N) == {True}
assert set(df_C.qname_C == df_M.qname_M) == {True}
df_Mf = df_M.drop(['qname_M'], axis=1)  # drop duplicated columns
df_Nf = df_N.drop(['qname_N'], axis=1)
dfs=[df_C, df_Mf, df_Nf]
for df in dfs:
    df.reset_index(drop=True, inplace=True)
df_en = pd.concat(dfs, ignore_index=True, axis=1)
df_en.columns = list(df_C.columns) + list(df_Mf.columns) + list(df_Nf.columns)
print 'Endotoxin domains table ({}): {}'.format(df_en.shape, df_en.columns)

## Collect physicochemical features homology -- pydpi ##
print 'Collecting physicochemical stats per protein...'
with open('./physicochem_annot_test.csv', 'w') as f:
    for record in SeqIO.parse(query, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        desc = protein.GetMoreauBrotoAuto()
        desc2 = protein.GetAAComp()
        z = desc.copy()
        z.update(desc2)
        len_p = str(len(record.seq))
        id = str(record.id)
        row = [id, len_p] + [str(i) for i in z.values()]
        f.write(','.join(row) + '\n')

df_de = pd.read_table('./physicochem_annot_test.csv', header=None, sep=',')
header = z.keys()
df_de.columns = ['id', 'seq_length'] + header
print 'Pydpi table({}): {}'.format(df_de.shape, df_de.columns)


## Build the master df with all three sets of data ##
# make protein names are sorted and matching
df_de = df_de.sort_values(by='id')
df_bl= df_bl.sort_values(by='qseqid')
df_en = df_en.sort_values(by='qname_C')
assert set(df_de['id'] == df_bl['qseqid']) == {True}
assert set(df_de['id'] == df_en['qname_C']) == {True}

# drop redundant cols and merge
df_en = df_en.drop(['qname_C'], axis=1)
df_bl = df_bl.drop(['qseqid'], axis=1)
dfs = [df_de, df_bl, df_en]
for df in dfs:
    df.reset_index(drop=True, inplace= True)
master_test = pd.concat(dfs, axis=1, ignore_index=True)
master_test.columns = list(df_de.columns) + list(df_bl.columns) + list(df_en.columns)

print 'Built master df ({}): {}'.format(master_test.shape, master_test.columns)
# Built master df ((10000, 287)): Index([u'id', u'seq_length', u'MoreauBrotoAuto_ResidueASA27',
#       u'MoreauBrotoAuto_ResidueASA26', u'MoreauBrotoAuto_ResidueASA25',
#       u'MoreauBrotoAuto_ResidueASA24', u'MoreauBrotoAuto_ResidueASA23',
#       u'MoreauBrotoAuto_ResidueASA22', u'MoreauBrotoAuto_ResidueASA21',
#       u'MoreauBrotoAuto_ResidueASA20',
#       ...
#       u'domain-score_M', u'reliability_M', u'count_M', u'e-value_N',
#       u'score_N', u'c-evalue_N', u'i-evalue_N', u'domain-score_N',
#       u'reliability_N', u'count_N'],
#      dtype='object', length=287)


master_test.to_pickle('putative_set.pkl')