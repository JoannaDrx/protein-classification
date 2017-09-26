"""
Second Genome - Genomic Data Scientist challenge
Author: Joanna Dreux [joanna.dreux@gmail.com]
"""

from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
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


def feature_selection(X, Y, feature_names, num):
    """ Feature selection with RFE and LinearSVC.
    Selects num features from X and returns the selected column names"""

    # logistic regression
    fit = RFE(LogisticRegression(), num).fit(X,Y)
    k = []
    for bool, ft in zip(fit.support_, feature_names):
        if bool:
            k.append(ft)
    print 'RFE-LR selected features:', k

    # linear SVC
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    n = []
    for bool, ft in zip(model.get_support(), feature_names):
        if bool:
            n.append(ft)
    print 'Linear SVC selected features:', n

    return list(set(k+n))


sel_features = feature_selection(X, Y, feature_names, 100)  # select for 100 features

# filter dfs on selected features
training = training_df[['id', 'label'] + sel_features]
test = test_df[['id'] + sel_features]

## Model selection ###
def validation_set(df, num):
    """ Creates Validation/testing datasets"""
    labels = df['label']
    features = df.ix[:, 2:].values
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(features, labels, test_size=num)

    return X_train, X_validation, Y_train, Y_validation


def compare_models(X_train, X_validation, Y_validation, Y_train):
    """ Compares accuracy of different models.
    Use 10x cross validation and use the same seed for comparison"""

    models = [('LR', LogisticRegression()), ('LinearSVC', LinearSVC()), ('KNN', KNeighborsClassifier()), ('SVC', SVC())]
    seed = 7
    # Compare Algorithms
    results = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        result = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(result)
        print '{}: avg = {}, (std = {})'.format(name, result.mean(), result.std())

    # plot results
    fig = plt.figure()
    plt.ylim([0.8, 1.001])
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(['LR', 'LinearSVC', "KNN", 'SVC'])
    plt.show()

    return


# make validation/training sets
xt, xv, yt, yv = validation_set(training, 0.2)
compare_models(xt, xv, yv, yt)


### Make predictions and score ###
def run_models(X_train, Y_train, test_array):

    # lr
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    p1 = lr.predict(test_array)
    s, m1 = score(p1)
    print 'LR: {}'.format(s)

    # knn
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    p2 = knn.predict(test_array)
    s, m2 = score(p2)
    print 'KNN: {}'.format(s)

    # lvc
    lvc = LinearSVC()
    lvc.fit(X_train, Y_train)
    p3 = lvc.predict(test_array)
    s, m3 = score(p3)
    print 'LinearSVC: {}'.format(s)

    return m1, m2, m3


def score(predictions):
    """ Score results using the dictionary"""

    pkl_file = 'protein_nomination_value.pkl'
    money = pickle.load(open(pkl_file, "rb"))

    df = pickle.load(open('putative_set.pkl', "rb"))

    df['label'] = predictions
    match = list(df[df['label'] == 1]['id'])

    # add up your $$$
    total = 0
    p_match = []
    for k, v in money.iteritems():
        if k in match:
            total += v
            p_match += k
    return total, p_match


p1, p2, p3 = run_models(training[sel_features], training['label'], test[sel_features])

# get all the predictions together
final = test['id']
final['p1'] = p1
final['p2'] = p2
final['p3'] = p3


### PLOTS ####
# plot the data
matches = training_df[training_df['label'] == 1]  # valid prots
non_matches = training_df[training_df['label'] == 0]  # invalid prots
fig, axes = plt.subplots(6, 4)
for ax, val in zip(axes.flatten(), [training_df['count_M'], training_df['count_N'],training_df['count_C']]):
    ax.hist(matches[val].values, alpha=0.5)
    ax.hist(non_matches[val].values, alpha=0.5)
    ax.set_title(val)
fig.set_tight_layout(True)

# PCA
X = training_df.ix[:,2:].values
y = training_df['label']
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print 'Pct variance explained by top 2 components: {}'.format(str(pca.explained_variance_ratio_))
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
