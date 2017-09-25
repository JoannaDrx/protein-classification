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
from sklearn import model_selection
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


def run_parse_blast(query, subject, threshold, out_file):

    # run blastp
    blastp_cline = NcbiblastpCommandline(query=query, subject=subject, evalue=threshold, outfmt=6, out=out_file)
    stdout, stderr = blastp_cline()

    # read & filter the blast results
    df = pd.read_table(out_file, header=None)
    blast_header = 'qseqid sseqid pident len mm gapopen qstart qend sstart send evalue bitscore'.strip().split(' ')
    df.columns = blast_header
    return df


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


def select_features(X, Y, num_features, feature_names):
    """Feature selection based on KBest, Logistic Regression and Trees Classifier
    Takes in array X of values, list Y of class and int for number of features wanted
    Returns the index of the features """

    # K Best
    test = SelectKBest(k=num_features)
    fit = test.fit(X, Y)
    k_features = unmask(fit.get_support(), feature_names)
    print 'Select K Best selected features:', k_features

    # Feature Extraction with RFE
    model = LogisticRegression()
    rfe = RFE(model, num_features)
    fit = rfe.fit(X, Y)
    rfe_features = unmask(fit.support_, feature_names)
    print 'RFE selected features:', rfe_features

    # Feature Importance with Extra Trees Classifier
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    tree_ft = []
    for i in range(num_features):
        tree_ft.append(feature_names[indices[i]])
    print 'Extra Trees Classifier selected features:', tree_ft

    return Counter(tree_ft + rfe_features + k_features)


def unmask(mask, names):
    """ Takes in mask and names, return names that are True"""
    l = []
    for bool, ft in zip(mask, names):
        if bool:
            l.append(ft)
    return l


def clean_up_hmm(df, suffix):
    """ For each df add a column counting the # of domains, then get the mean to collapse into single row/query"""

    df = df.drop(['target', 'acc'], axis=1)
    df = df.groupby('qname', as_index=False).mean()  # average out multiple domains
    df = df.add_suffix('_' + suffix)
    print df.shape
    print df.columns

    return df
