"""
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
from sklearn import preprocessing

# Import the pickled master dfs
training_df = pickle.load(open('training_set.pkl', "rb"))
test_df = pickle.load(open('putative_set.pkl', "rb"))


# plot the data
X = training_df.ix[:,2:].values
y = training_df['label']
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
plt.figure()
for color, i in zip(['navy', 'darkorange'], [0, 1]):
    plt.scatter(X_r[y[y == i].index, 0], X_r[y[y == i].index, 1], color=color, alpha=.8, lw=2, label=str(i))
plt.legend(loc='best', shadow=False, scatterpoints=1)

# data normalization
X_scaled = preprocessing.scale(X)
# plot the normalized data
X_rs = pca.fit(X_scaled).transform(X_scaled)
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_rs[y[y == i].index, 0], X_rs[y[y == i].index, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)


### Feature Selection ###
feature_names = list(training_df.columns[2:])

def feature_selection(X, Y, feature_names, num):
    """ Feature selection with RFE and LinearSVC.
    Selects num features from X and returns the selected column names"""

    # logistic regression
    fit = RFE(LinearSVC(), num).fit(X,Y)
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


sel_features = feature_selection(X, y, feature_names, 200)  # select for 200 features

# filter dfs on selected features
training = training_df[['id', 'label'] + sel_features]
test = test_df[['id'] + sel_features]

# plot PCA after feature selection
X = training.ix[:,2:].values
y = training['label']
X_scaled = preprocessing.scale(X)
# plot the normalized data
X_rs = pca.fit(X_scaled).transform(X_scaled)
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_rs[y[y == i].index, 0], X_rs[y[y == i].index, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)




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
xt, xv, yt, yv = validation_set(training, 0)
compare_models(xt, xv, yv, yt)

### Make predictions and score ###
def run_models(X_train, Y_train, test_array):

    # knn
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    p1 = knn.predict(test_array)
    print 'KNN: {}'.format(score(p1))

    # lvc
    lvc = LinearSVC()
    lvc.fit(X_train, Y_train)
    p2 = lvc.predict(test_array)
    print 'LinearSVC: {}'.format(score(p2))

    return p1, p2


def score(predictions):
    """ Score results using the dictionary"""

    pkl_file = 'protein_nomination_value.pkl'
    money = pickle.load(open(pkl_file, "rb"))

    df = pickle.load(open('putative_set.pkl', "rb"))

    df['label'] = predictions
    match = list(df[df['label'] != 0]['id'])

    # add up your $$$
    total = 0
    for k, v in money.iteritems():
        if k in match:
            total += v
    return total


p1, p2 = run_models(training[sel_features], training['label'], test[sel_features])


## plot the putative set predictions on PCA
df_p1 = test.assign(label=p1)
df_p2 = test.assign(label=p2)

for df in [df_p1, df_p2]:
    X = df.ix[:,1:-1].values
    y = df['label']
    pca = PCA(n_components=2)
    X = preprocessing.scale(X)
    X_r = pca.fit(X).transform(X)
    plt.figure()
    for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
        plt.scatter(X_r[y[y == i].index, 0], X_r[y[y == i].index, 1], color=color, alpha=.8, lw=2,label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)


## peek at ideal results
pkl_file = 'protein_nomination_value.pkl'
money = pickle.load(open(pkl_file, "rb"))

best = sorted(money.keys())
l = []
for i in test['id']:
    if i in best:
        l.append(1)
    else:
        l.append(0)

t= test.assign(label=l)
X = t.ix[:,1:-1].values
y = t['label']
pca = PCA(n_components=2)
X = preprocessing.scale(X)
X_r = pca.fit(X).transform(X)
plt.figure()
for color, i, target_name in zip(['navy', 'darkorange'], [0, 1], ['0', '1']):
    plt.scatter(X_r[y[y == i].index, 0], X_r[y[y == i].index, 1], color=color, alpha=.8, lw=2,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)

