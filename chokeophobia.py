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
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pydpi.pypro import CTD, AAComposition, PyPro
import pickle

# cwd
os.chdir('/Users/jdreux/PycharmProjects/protein-classification/')

## Gather & eyeball the data

# import the data
training_set = 'drugpro_training_set.csv'
names = ['id', 'seq', 'label']
ts = pd.read_csv(training_set, header=0)
print ts.shape
print ts.head(20)
ts.groupby('label').size()
# label
# 0    220
# 1     44

# 1. BLAST - sequence homology
subject = 'training_seqs.fasta'
out_blastp = 'blastp_out.tab'

# Convert training set to FASTA file format, annotate with label
seq_list = []
for id, seq, label in ts.itertuples(index=False):
        seq_record = SeqRecord(Seq(seq, IUPAC.protein), id=id+str("-label:"+str(label)),
                               description="label: "+str(label), name="")
        seq_list.append(seq_record)
handle = open(subject, "w")
for sequences in seq_list:
    SeqIO.write(sequences, handle, "fasta")

# blastp against the training set
blastp_cline = NcbiblastpCommandline(query=subject, subject=subject, evalue=1e-7, outfmt=6, out=out_blastp)
stdout, stderr = blastp_cline()

# read & filter the results
df = pd.read_table(out_blastp, header=None)
outfmt6_cols = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
df.columns = outfmt6_cols

# filter for pident >= 90.0 && remove identical hits
df_filtered = df[(df['pident'] >= 90.0) & (df['pident'] != 100.0)]

# check if any homologies between 0 and 1
false_pos = df_filtered[(df_filtered['sseqid'].str.contains('label:1')) & (df_filtered['qseqid'].str.contains('label:0'))]
false_pos.describe()

# positive matches  -- 82 seqs
matches = df_filtered[(df_filtered['sseqid'].str.contains('label:1')) & (df_filtered['qseqid'].str.contains('label:1'))]
matches.describe()

# pident consistently high in matches
# len is strong predictor
# mm bad predictor, drop gapopen, qstart, qend
# sstart is useful < 28


# 2. Hidden Markov Models -- domain similarity
out_hmmer = 'out_hmmer-training-set.txt'
pfam_db = '/Users/jdreux/Desktop/hmm-db/Pfam-A.hmm'
threshold = 1e-5

# against training set, find main domains
cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, subject)
os.system(cmd)

# read the results, select relevant cols
df2 = pd.read_table(out_hmmer, header=None, engine='python', skiprows=3, skipfooter=10, sep='\s+',
                    usecols=[0,1,3,5,6,7,11,12,13, 15, 16, 17, 18, 19, 20, 21])
header = 'target acc qname qlen e-value score cond-e-value ind-e-value domain-score hf ht af at ef et reliability'.strip().split(' ')
df2.columns = header

# eyeball the valid sequences
pos_matches = df2[df2['qname'].str.contains('label:1')]
len(set(pos_matches['qname']))  # check they're all represented

motifs_of_interest = set(pos_matches['acc'])  # select few represented in valid prots

# does each query have all 3?
set(pos_matches['qname'].value_counts())
not_all_three = []
for qname in set(pos_matches['qname']):
    tmp_df = pos_matches[pos_matches['qname'] == qname]
    if len(set(tmp_df['acc'])) < 3:
        not_all_three.append(qname)

# not_all_three == [] -- every valid prot has all 3

# for each find out the parameters
# delta endotoxin C
endo_C = pos_matches[pos_matches['acc'] == 'PF03944.13']
endo_C.describe()

# delta endotoxin M
endo_M = pos_matches[pos_matches['acc'] == 'PF00555.18']
endo_M.describe()

# delta endotoxin N
endo_N = pos_matches[pos_matches['acc'] == 'PF03945.13']
endo_N.describe()


# eyeball the label = 0 prots
neg_matches = df2[df2['qname'].str.contains('label:0')]
neg_matches = neg_matches[neg_matches['acc'].str.contains('PF03945.13|PF00555.18|PF03944.13')]

# does each query have all 3?
set(neg_matches['qname'].value_counts())  # they can have 1,2,3 or 4 of the relevant domains


# focus on those that have 3-4 (at least one of each)
enough_motifs = neg_matches.groupby('qname').filter(lambda x: len(x) > 2)  # narrow to 139

# then check they have one of each
not_all_three = []
for qname in set(enough_motifs['qname']):
    tmp_df = enough_motifs[enough_motifs['qname'] == qname]
    if len(set(tmp_df['acc'])) < 3:
        not_all_three.append(qname)
# drop em if not
not_in = enough_motifs[~enough_motifs['qname'].isin(not_all_three)]

endo_Cneg = not_in[not_in['acc'] == 'PF03944.13']
endo_Cneg.describe()

endo_Mneg = not_in[not_in['acc'] == 'PF00555.18']
endo_Mneg.describe()

endo_Nneg = not_in[not_in['acc'] == 'PF03945.13']
endo_Nneg.describe()


# 3. Physicochemical features homology using pydpi
annotation_csv = 'physicochem_annot.csv'
# go through fasta and look at a AA composition, start writing annotation table
with open(annotation_csv, 'w') as f:
    annot_header = 'id,Label,A,C,E,D,G,F,I,H,K,M,L,N,Q,P,S,R,T,W,V,Y\n'
    f.write(annot_header)
    for record in SeqIO.parse(subject, "fasta"):
        protein = PyPro()
        protein.ReadProteinSequence(str(record.seq))
        aas = protein.GetAAComp()
        id, label = str(record.id).strip().split('-label:')
        f.write(id + ',' + label + ',' + ','.join([str(i) for i in aas.values()]) + '\n')

aas = pd.read_table(annotation_csv, header=0, sep=',')


# plot it out
for amino in annot_header.split(',')[2:]:
    x = aas[aas['Label'] == 1][amino]
    y = aas[aas['Label'] == 0][amino]
    bins = np.linspace(0, 20, 5)
    plt.hist(x, bins, alpha=0.5, label='Label 1')
    plt.hist(y, bins, alpha=0.5, label='Label 0')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

# Norm
x = ctd[ctd['Label'] == 1]["Normalized"]
y = ctd[ctd['Label'] == 0]["Normalized"]
bins = np.linspace(65, 85, 10)
plt.hist(x, bins, alpha=0.5, label='x')
plt.hist(y, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
plt.close()


x = ctd[ctd['Label'] == 1]["Polarity1"]
y = ctd[ctd['Label'] == 0]["Polarity1"]
bins = np.linspace(65, 85, 10)
plt.hist(x, bins, alpha=0.5, label='x')
plt.hist(y, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
plt.close()


x = ctd[ctd['Label'] == 1]["Charge"]
y = ctd[ctd['Label'] == 0]["Charge"]
bins = np.linspace(15, 30, 10)
plt.hist(x, bins, alpha=0.5, label='x')
plt.hist(y, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
plt.close()

plt.hist(ctd["Polarity2"])
plt.show()

plt.hist(ctd["SecondaryStr"])
plt.show()


x = ctd[ctd['Label'] == 1]["SecondaryStr"]
y = ctd[ctd['Label'] == 0]["SecondaryStr"]
bins = np.linspace(15, 30, 10)
plt.hist(x, bins, alpha=0.5, label='x')
plt.hist(y, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
plt.close()


# PCA here works, color by label

from sklearn.cluster import KMeans

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=2, random_state=1)
# Get only the numeric columns from games.
good_columns = ctd._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
plt.close()
# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.
plt.show()



## figure out what to predict
#There are two things we need to determine before we jump into machine learning – how we’re going to measure error,
## and what we’re going to predict. We thought earlier that average_rating might be good to predict on, and our
# exploration reinforces this notion.

#There are a variety of ways to measure error (many are listed here). Generally, when we’re doing regression,
# and predicting continuous variables, we’ll need a different error metric than when we’re performing classification,
# and predicting discrete values.

#For this, we’ll use mean squared error – it’s easy to calculate, and simple to understand. It shows us how far, on
# average, our predictions are from the actual values.

games.corr()['average_rating']

# values closest to 1 are what you want



## pick predictor columns

# Before we get started predicting, let’s only select the columns that are relevant when training our algorithm.
# We’ll want to remove certain columns that aren’t numeric.

#We’ll also want to remove columns that can only be computed if you already know the average rating. Including
# these columns will destroy the purpose of the classifier, which is to predict the rating without any previous
# knowledge. Using columns that can only be computed with knowledge of the target can lead to overfitting, where
# your model is good in a training set, but doesn’t generalize well to future data.

# Get all the columns from the dataframe.
columns = games.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]

# Store the variable we'll be predicting on.
target = "average_rating"



## split into trina and set

#We want to be able to figure out how accurate an algorithm is using our error metrics. However, evaluating the
# algorithm on the same data it has been trained on will lead to overfitting. We want the algorithm to learn
# generalized rules to make predictions, not memorize how to make specific predictions. An example is learning
# math. If you memorize that 1+1=2, and 2+2=4, you’ll be able to perfectly answer any questions about 1+1 and 2+2.
# You’ll have 0 error. However, the second anyone asks you something outside of your training set where you know
# the answer, like 3+3, you won’t be able to solve it. On the other hand, if you’re able to generalize and learn
# addition, you’ll make occasional mistakes because you haven’t memorized the solutions – maybe you’ll get
# 3453 + 353535 off by one, but you’ll be able to solve any addition problem thrown at you.


#If your error looks surprisingly low when you’re training a machine learning algorithm, you should always
# check to see if you’re overfitting.

#In order to prevent overfitting, we’ll train our algorithm on a set consisting of 80% of the data, and test
# it on another set consisting of 20% of the data. To do this, we first randomly samply 80% of the rows to be
# in the training set, then put everything else in the testing set.

# Import a convenience function to split the sets.
from sklearn.cross_validation import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)



#Linear regression is a powerful and commonly used machine learning algorithm. It predicts the target variable using
# linear combinations of the predictor variables. Let’s say we have a 2 values, 3, and 4. A linear combination would
# be 3 * .5 + 4 * .5. A linear combination involves multiplying each number by a constant, and adding the results.
# You can read more here.

#Linear regression only works well when the predictor variables and the target variable are linearly correlated.
# As we saw earlier, a few of the predictors are correlated with the target, so linear regression should work well for us.

#We can use the linear regression implementation in Scikit-learn, just as we used the k-means implementation earlier.

# Import the linearregression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

#When we fit the model, we pass in the predictor matrix, which consists of all the columns from the dataframe that
# we picked earlier. If you pass a list to a Pandas dataframe when you index it, it will generate a new dataframe
# with all of the columns in the list. We also pass in the target variable, which we want to make predictions for.

#The model learns the equation that maps the predictors to the target with minimal error.


# After we train the model, we can make predictions on new data with it. This new data has to be in the exact same
# format as the training data, or the model won’t make accurate predictions. Our testing set is identical to the
# training set (except the rows contain different board games). We select the same subset of columns from the test
# set, and then make predictions on it.

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test[target])

## try random forest
#One of the nice things about Scikit-learn is that it enables us to try more powerful algorithms very easily. One
# such algorithm is called random forest. The random forest algorithm can find nonlinearities in data that a linear
# regression wouldn’t be able to pick up on. Say, for example, that if the minage of a game, is less than 5, the
# rating is low, if it’s 5-10, it’s high, and if it is between 10-15, it is low. A linear regression algorithm
# wouldn’t be able to pick up on this because there isn’t a linear relationship between the predictor and the target.
# Predictions made with a random forest usually have less error than predictions made by a linear regression.

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])



##### Get some result


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





##
# against ours
# we build a custom HMM refrence from our three domains of interest obtained at: http://pfam.xfam.org/
# concatenate all three HMM files for these endotoxin families

pfam_db = 'endotoxins-hmm/endotoxins.hmm'

cmd = 'hmmscan --domE {} --domtblout {} --noali --cpu 8 {} {} > /dev/null'.format(threshold, out_hmmer, pfam_db, query)
os.system(cmd)


# look for domains with the lowest e-values and highest bit scores
# narrow down to about 961 seqs










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






# build list of seqs from the whole set
query_list = []
for record in SeqIO.parse(query, "fasta"):
    query_list.append(record.id)