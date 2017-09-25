"""
Second Genome - Genomic Data Scientist challenge
Author: Joanna Dreux [joanna.dreux@gmail.com]
"""

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE, RFECV
from sklearn import model_selection
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle


# Import the pickled master dfs
training_df = pickle.load(open('training_set.pkl', "rb"))
test_df = pickle.load(open('putative_set.pkl', "rb"))


### Feature Selection ###
array = training_df.values
X = array[:,2:]
Y = list(array[:,1])  # labels
feature_names = list(training_df.columns[2:])


def feature_sel(X, Y, feature_names, num):

    # Feature Extraction with RFE
    model = LogisticRegression()
    rfe = RFE(model, num)
    fit = rfe.fit(X, Y)
    k = []
    for bool, ft in zip(fit.support_, feature_names):
        if bool:
            k.append(ft)
    print 'RFE selected features:', k

    # Feature Extraction with RFE
    model2 = SVC(kernel="linear")
    rfe = RFE(model2, num)
    fit = rfe.fit(X, Y)
    m = []
    for bool, ft in zip(fit.support_, feature_names):
        if bool:
            m.append(ft)
    print 'RFE-SVC selected features:',m

    # Feature Extraction with RFE
    model3 = LinearSVC()
    rfe = RFE(model3, num)
    fit = rfe.fit(X, Y)
    n = []
    for bool, ft in zip(fit.support_, feature_names):
        if bool:
            n.append(ft)
    print 'RFE-LinearSVC selected features:',n

    # Feature Extraction with RFE
    model4 = SVC(kernel='linear')
    rfe = RFECV(model4, num, scoring='accuracy')
    fit = rfe.fit(X, Y)
    o = []
    for bool, ft in zip(fit.support_, feature_names):
        if bool:
            o.append(ft)
    print 'RFECV selected features:',o


    return k, m, n, o


def validation_set(df, num):
    # Create Validation Dataset
    labels = df['label']
    features = df.ix[:, 2:].values
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=num)

    return X_train, X_validation, Y_train, Y_validation


## Try out multiple models
models = [('LR', LogisticRegression()), ('LinearSVC', LinearSVC()), ('KNN', KNeighborsClassifier())]


def run_models(models, X_train, X_validation, Y_validation, Y_train, test_array, seed=7, scoring='accuracy'):
    ## Use 10 fold cross validation to estimate accuracy of each model tested
    # seed: using same random seed to make results comparable
    # scoring ratio correct predictions: total predictions * 100

    # Compare Algorithms
    #results = []
    #for name, model in models:
    #    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #    result = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #    results.append(result)
    #    print '{}: {}, ({})'.format(name, result.mean(), result.std())
    #print ('#########################')
    ### Make predictions

    # knn
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(test_array)
    score(predictions)
    #print accuracy_score(Y_validation, predictions)
    #print confusion_matrix(Y_validation, predictions)
    #print classification_report(Y_validation, predictions)

    # logistic regression
    lin = LogisticRegression()
    lin.fit(X_train, Y_train)
    predictions2 = lin.predict(test_array)
    score(predictions2)
    #print accuracy_score(Y_validation, predictions)
    #print confusion_matrix(Y_validation, predictions)
    #print classification_report(Y_validation, predictions)

    # lvc
    lvc = LinearSVC()
    lvc.fit(X_train, Y_train)
    predictions = lvc.predict(test_array)
    score(predictions)
    #print accuracy_score(Y_validation, predictions)
    #print confusion_matrix(Y_validation, predictions)
    #print classification_report(Y_validation, predictions)

    return


def score(predictions):
    """ Score results using the dictionary"""

    pkl_file = 'protein_nomination_value.pkl'
    money = pickle.load(open(pkl_file, "rb"))

    df = pickle.load(open('putative_set.pkl', "rb"))

    df['label'] = predictions
    match = list(df[df['label'] == 1]['id'])

    # add up your $$$
    total = 0
    for k, v in money.iteritems():
        if k in match:
            total += v
    print "{}".format(total)

    return total


k,m,n,o = feature_sel(X, Y, feature_names, 100)
# filter master
mk = training_df[['id', 'label'] + k]
mm = training_df[['id', 'label'] + m]
mn = training_df[['id', 'label'] + n]
mo = training_df[['id', 'label'] + o]
# filter test
tk = test_df[['id'] + k]
tm = test_df[['id'] + m]
tn = test_df[['id'] + n]
to = test_df[['id'] + o]


# valid sets
xt2, xv2, yt2, yv2 = validation_set(mk, 0)
print '######Looking at 2'
run_models(models, xt2, xv2, yv2, yt2, tk.values[:, 1:])

xt1, xv1, yt1, yv1 = validation_set(mm, 0)
xt2, xv2, yt2, yv2 = validation_set(mn, 0)
print '#####Looking at 3'
run_models(models, xt1, xv1, yv1, yt1, tm.values[:, 1:])
print '######Looking at 4'
run_models(models, xt2, xv2, yv2, yt2, tn.values[:, 1:])

xt1, xv1, yt1, yv1 = validation_set(mo, 0)
print '#####Looking at 5'
run_models(models, xt1, xv1, yv1, yt1, to.values[:, 1:])


sum(money.values())
580000000.0
######Looking at 2
2000000.0
4000000.0
64000000.0
#####Looking at 3
2000000.0
4000000.0
64000000.0
######Looking at 4
14000000.0
0
10000000.0
#####Looking at 5
2000000.0
4000000.0
60000000.0




### PLOTS ####
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


