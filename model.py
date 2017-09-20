"""
Model for training
How is this even supposed to look
"""

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://www.datacamp.com/community/tutorials/machine-learning-python#gs.AsFJLjQ
# https://github.com/mhorlbeck/CRISPRiaDesign/blob/master/Library_design_walkthrough.md
# https://pypi.python.org/pypi/pydpi/1.0
# http://scikit-bio.org/docs/0.1.4/core.alignment.html

import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# import the training set
training_set = '/Users/joanna/Desktop/SG_GDS_Challenge_092017 2/drugpro_training_set.csv'
names = ['id', 'seq', 'label']
ts = pd.read_csv(training_set, header=0)

# check out the data
print ts.shape
print ts.head(20)
ts.groupby('label').size()

# plots here
plt.hist(ts["label"])
plt.show()

# plot base position colored by label

# pca here?


## 1. Sequence homology

pct = 0.9  # set at 90% homology
cmd = './vsearch --usearch_global queries.fsa --db database.fsa --id {} --alnout alnout.txt'.format(pct)

 # > NCBI's blastp:
 #   https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins

 # > vsearch - a fast heuristic-based sequence identity search
#    https://github.com/torognes/vsearch

#There are many others. If you choose to leverage sequence identity in your
#method, feel free to use whichever tool you deem most appropriate.

# 2. Hidden Markov Models -- domain similarity

#In addition to sequence identity, hidden-markov models are well known to detect
#"remote hology", that is, a functional relationship between two proteins whose
#sequence identity may be very low.  The most-often used tool for scanning
#protein sequences against a know set of HMMs is hmmer:

#  > HMMER
#    http://hmmer.org/

#A typical source of HMMs with which to annotate proteins using HMMER is the
# Pfam database of protein domains:    http://pfam.xfam.org/

# 3. Machine Learning to classify based on primary sequence

# machine learning models trained on physicochemical features to classify proteins of unknown function from their
# primary sequence.  There is no such thing as "the" set of physicochemcial
# features for a particular protein.  The set is effectively infinite.  However,
# there are several resources for calculating a fairly comprehensive set of
# physicochemcial features from a set of sequences.  They are:

#  > ProFEAT (webserver)
#    http://bidd2.nus.edu.sg/cgi-bin/profeat2016/main.cgi

#  > ProtrWeb (R)
#    http://protrweb.scbdd.com/

#  > Pydpi (Python)
#    https://pypi.python.org/pypi/pydpi/1.0

# 4. Extras

# numeric components
# length, AA content,

from sklearn.cluster import KMeans

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_


# add whatever features you deem necissary from the
# primary sequences to accomplish the classification task, but these should
# provide a sufficient start.

#The way I would attack the problem, in general, is to leverage statistical analysis like Principal Component
# Analysis or Ordinary Least Squares to help determine what attributes within these protein sequences are best
# suited to classify proteins as Neuropeptide hormone precursors.

#In order to do that, you'll have to convert the protein sequences into numeric data, but I believe some work
# has already been done in that regard using formulas leveraged in Amino Acid PCA.

#See these two links: http://www.ncbi.nlm.nih.gov/pubmed/24496727

#http://www.ncbi.nlm.nih.gov/pubmed/16615809

#Once that work has been done, I would attempt to classify using the entire dataset and a reinforcement
# learning algorithm, like Naive Bayes while slimming down the data into that which PCA has identified as important.


### Evaluation  ###

#DrugPro has agreed to synthesize and test 200 proteins of your choice in their
#assay.  Because DrugPro is concerned about whether they can obtain legal rights
#to well-known therapies for chokeophobia, you will be remunerated for proteins
#that score positively in their assay according to how similar they are to
#existing chokeophobia therapeutics in their training set:

 #   >= 78% sequence identity: $1M per protein
  #  < 78% but >= 45% sequence identity: $2M per protein
   # < 45% sequence identity: $4M per protein

#Normally it requires several months to synthesize and test a protein, so we
#will be accelerating that process with the following validation file:

#  > protein_nomination_value.pkl

#Once you are done training your classifier and applying it to the putative
#set of 10000 proteins, use this pickled dictionary to report the monetary
#value returned by your Top 200 nominations.  Obviously you could pretty
#easily use this dictionary to just go and search for high-valued proteins
#directly in the set of 10000, but that wouldn't make for a very good presentation.

########
# code documentation
# communicaiton plan
# powerpoint slides

#  > a brief description of the problem (it helps to hear your version of it)
#  > a brief description of the data
#  > your choice of features engineered from the protein sequences
#  > your feature selection methodolgy, if any
#  > your choice of model or models
#  > your validation method(s)
#  > the quality of your model ($$$)


#Define Problem.
#Prepare Data.
#Evaluate Algorithms.
#Improve Results.
#Present Results.

