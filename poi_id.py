#!/usr/bin/python
# coding=utf-8


import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../tools/")
sys.path.append("../final_project")

from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data, test_classifier


from tester import dump_classifier_and_data

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


from numpy import matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import CountVectorizer

# Validation

from sklearn import cross_validation

# Feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

# Classification

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import grid_search, datasets
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

'''
Task 1: Select what features I will use for the best result and why.

Below is all features currently I have, then What features should I use to get a best result.
What is the difference of the result between when using two features and all features in precision or recall
'''


features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus','restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other','long_term_incentive',
                 'restricted_stock', 'director_fees']


'''
Find who made a fraud from enron employee data set.
which columns do I need to use!! there are too many columns.
Find a relation between features. What are the most related features with fraud.

1. Select features according to the k highest value (SelectKbest)
2. PCA
3. pipeline

But firstly,
Load the dictionary containing the data set
Open pk data file
file format = pkl, dictionary type of data set
'''


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


'''
# Read all people name and each feature values with dictionary type
# The method items() returns a list of dict's (key, value) tuple pairs
'''

name_list = [key for key, value in data_dict.items()]


# remove a name who had no information inside, It will be automatically removed in featureFormat function
name_list.remove('LOCKHART EUGENE E')

'''
featureFormat
change data set to multi-dimensional numpy array with feature values.
ex) array([[ -1.13698227e+002,   4.25087011e-303],[  2.88528414e-306,   3.27025015e-309]])
'''

data = featureFormat(data_dict, features_list)

data_pd = pd.DataFrame(data, index=name_list, columns=features_list)

# Just want to check how many zeros in each features
data_zero_list = (data_pd == 0).sum(axis=0)/len(data_pd)
data_zero_list.sort_values(axis=0, ascending = True)

'''
zero values included in each column

features                 portion of zeros
total_stock_value            0.131034
total_payments               0.137931
restricted_stock             0.241379
exercised_stock_options      0.296552
salary                       0.344828
expenses                     0.344828
other                        0.358621
bonus                        0.434483
long_term_incentive          0.544828
deferred_income              0.662069
deferral_payments            0.731034
poi                          0.875862
restricted_stock_deferred    0.875862
director_fees                0.882759
loan_advances                0.972414

loan_advances, director_fees almost useless, most of them are zeros.
Let's try to combine each other to make some useful features,
If I could understand what each feature exactly mean on financial term, I might get a better result
'''


'''
Added new features using current data set.
a.	data_pd["TotalStock_TotalPay"] = data_pd["total_stock_value"] + data_pd["total_payments"]
#It had “Total” in common. They might have a relationship if total stock value is high then total payments might be high as well or not.

b.	data_pd["Salary_Bonus"] = data_pd["salary"] + data_pd["bonus"]
# It is the genuine value of the employee earned in a year including all the money someone received from the company.
We normally think yearly income as salary and bonus together.

c.	data_pd["TotalStock_RestStock_ExercStock"] = data_pd["total_stock_value"] + data_pd["restricted_stock"] + data_pd["exercised_stock_options"]
# Added all the stock related values
'''

data_pd["TotalStock_TotalPay"] = data_pd["total_stock_value"] + data_pd["total_payments"]
data_pd["Salary_Bonus"] = data_pd["salary"] + data_pd["bonus"] # Seems like genuine salary in a year
# data_pd["Total"] = data_pd.sum(axis = 1) - data_pd["deferred_income"]
data_pd["TotalStock_RestStock_ExercStock"] = data_pd["total_stock_value"] + data_pd["restricted_stock"] + data_pd["exercised_stock_options"] # All the stock related values


'''
Let's calculate zero values again to figure out what changed
'''

data_zero_list = (data_pd == 0).sum(axis=0)/len(data_pd)
data_zero_list.sort_values(axis=0, ascending = True)

'''
zero values included in each column

features                      portion of zeros
TotalStock_TotalPay                0.013793
TotalStock_RestStock_ExercStock    0.117241
total_stock_value                  0.131034
total_payments                     0.137931
restricted_stock                   0.241379
exercised_stock_options            0.296552
Salary_Bonus                       0.344828
expenses                           0.344828
salary                             0.344828
other                              0.358621
bonus                              0.434483
long_term_incentive                0.544828
deferred_income                    0.662069
deferral_payments                  0.731034
restricted_stock_deferred          0.875862
poi                                0.875862
director_fees                      0.882759
loan_advances                      0.972414
'''

'''
Tried to add text learning but stuck at some points
Please give me some advice.


what I think how to handle this was to find frequently used words between people who are POI == 1, and
apply those words to the others, who are suspected, how many times do they use

but I couldn't find poi == 1 names email in maildir.
'''


''''''''''''''''''''''''
# Eliminates outliers from the features
''''''''''''''''''''''''

data_pd[['TotalStock_TotalPay','poi']].boxplot(by = 'poi')

# I could see there are several outliers exist
print data_pd[['TotalStock_TotalPay','poi']].sort_values(by = ['TotalStock_TotalPay'])

'''
When I look through the plot. one outlier was in all the plots. It might be a person or might be several people
Let's remove the outlier. I am not sure which data was out of other points, different from the other values.
but when I look through one coloumn. There was "total" row in data set
Drop the total feature, which is outlier. too higher than the other values.
'''


# Found there is 'TOTAL, LAY KENNETH L' in index name. let's drop them both
data_pd.drop(['TOTAL'], inplace=True)

# data_pd_new will be used for standardize later using total value without dropping it
data_pd_new = data_pd.copy()

# data_pd.drop(['LAY KENNETH L'], inplace=True)

'''
Needed to keep the POI data as well as needed to find the reasonable outliers, which could make me not to remove most of lists below.
'''
row_zero_list = ((data_pd==0).astype(int).sum(axis=1)) # Row NaN found

'''
                        numbers of zero values
WODRASKA JOHN                    15
POWERS WILLIAM                   16
THE TRAVEL AGENCY IN THE PARK    15
BROWN MICHAEL                    15
GRAMM WENDY L                    15

They have too many zero values inside the same as "THE TRAVEL AGENCY IN THE PARK" which seems like reasonable to remove.
'''

for a in data_pd.index:
    if a in row_zero_list[row_zero_list>14].index:
        # print  data_pd.index[a] % data_pd.loc[a]['poi']
        print "%s is POI %d" %(a, data_pd.loc[a]['poi'])
# They are all zero poi value. I think they don't that much affect on searching criminal. POI is zero for them
# remove them from the list

data_pd.drop(row_zero_list[row_zero_list>14].index, inplace =True)

data_pd[['TotalStock_TotalPay','poi']].boxplot(by = 'poi')


total_list_zero = data_pd[data_pd['poi'] == 0].sort_values(by = 'TotalStock_TotalPay', ascending = False)
total_list_one = data_pd[data_pd['poi'] == 1].sort_values(by = 'TotalStock_TotalPay', ascending = False)
total_list_one = total_list_one.index.tolist()
total_list_zero = total_list_zero.index.tolist()


data_original = data_pd.copy()


'''
Let's try SelectKbest first. Select features according to the k highest scores.
Kbest feature is good for if I know how many features needed to find
'''

fold = 1000

from sklearn.cross_validation import StratifiedShuffleSplit

splits = StratifiedShuffleSplit(data_pd['poi'], fold, test_size=0.1, random_state=43)
k = len(list(data_pd)) - 1
# k = 5
# print splits
# We will include all the features into variable best_features, then group by their
# occurrences.
features_list = list(data_pd.drop(['poi'], axis=1))
features = data_pd.drop(['poi'], axis=1).as_matrix()
labels = data_pd['poi'].as_matrix()

print features_list
# print features[1]
best_features = []
best_scores = []

features_train = []
labels_train = []
features_test = []
labels_test = []

for i_train, i_test in splits:
    for ii in i_train:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in i_test:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

#     print features_train
    # fit selector to training set
    selector = SelectKBest(k = k)
    selector.fit(features_train, labels_train)

#     print selector.scores_
#     print selector.get_support(indices = True)

    for i, j in zip(selector.get_support(indices = True), selector.scores_):
        best_features.append(features_list[i])
        best_scores.append(j)
# print best_features


from collections import defaultdict
d = defaultdict(int)

for idx, key in enumerate(best_scores):
    if idx > k-1:
        idx = idx % k

    d[idx] += key

#     print d
# for i in best_features:
#     d[i] += 1

print d

import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)

print sorted_d
features_list = [ features_list[x[0]] for x in sorted_d[-k:]]

print features_list
# print features_list[1:]

# print [ i[1] for i in sorted_d[-k:][1:]]

# SelectionKBest score total
KBest_scores = pd.DataFrame({'Score' : [ i[1] for i in sorted_d[-k:]]}, index = features_list)

# print KBest_scores.index
print KBest_scores



'''
                                Accumulated Score
exercised_stock_options          1.068181e+07
total_stock_value                1.039250e+07
TotalStock_RestStock_ExercStock  1.030532e+07
Salary_Bonus                     9.650207e+06
bonus                            8.893023e+06
salary                           7.707642e+06
TotalStock_TotalPay              7.352396e+06
deferred_income                  4.944756e+06
long_term_incentive              4.165918e+06
restricted_stock                 3.908955e+06
total_payments                   3.821471e+06
loan_advances                    3.177775e+06
expenses                         2.575345e+06
other                            1.815894e+06
director_fees                    8.900266e+05
deferral_payments                1.151085e+05
restricted_stock_deferred        3.080609e+04
'''

'''
* STUDY
cross_validation - k_folds, shuffle, etc...

* compare the result with selectKbest features and using all features

* scaling features how could it affect on the result

'''

# color_form = ["r", "b"]
#
# plt.figure(3)
#
# for index, ii in data_pd.iterrows():
#
#     poi = int(ii['poi'])
#     ttl_stck_vl = ii['Total']
#     xrcs_stck_ptns = ii['restricted_stock_deferred']
#
#     plt.scatter(ttl_stck_vl, xrcs_stck_ptns, color = color_form[poi])

'''''''''''''''''''''''''''''''''''''''''''''
Just wondering even though I found the most related features, I couldn't figure out with plots.
don't understand why they are the most influential features.
'''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''
Do I need to back to generate training and test set? or ?
'''''''''''''''''''''''''''''''''''''''''''''
pp_nm = list(data_pd.index)
pp_cl = list(data_pd.columns)

cv = StratifiedShuffleSplit(data_pd['poi'], fold, test_size=0.1, random_state=43)

#
# dataset = data_pd.to_dict(orient="index")
# feature_list = ['poi'] + features_list
def data_set(var):
    data_poi = data_pd['poi']
    data_nonpoi = data_pd.drop(['poi'], axis = 1)

    if var == 'log':
        data_nonpoi = np.log10(data_nonpoi-min(data_pd.min(axis = 0))+1)
        data_nonpoi['poi'] = data_poi
#     data_pd = data_nonpoi
#     data_pd = data_poi + data_nonpoi
    elif var == 'none':
        data_nonpoi = data_pd
    elif var == 'sqrt':
        data_nonpoi = np.sqrt(data_nonpoi-min(data_pd.min(axis = 0)))
        data_nonpoi['poi'] = data_poi
    elif var == 'mnmx':
        scaler = MinMaxScaler()
        dt = scaler.fit_transform(data_pd)
        dt = pd.DataFrame(dt, index=pp_nm, columns=pp_cl)
        data_nonpoi = dt

    return data_nonpoi


def Gaussian_compare():

# GaussianNB
    pipeline = Pipeline([
                        ('select', SelectKBest()),
                        ('pca', PCA()),
                         ('clf', GaussianNB())
                        ])
    params = {
            'select__k' : range(15,18),
             'pca__n_components' : range(6,10)
            }

    clf = grid_search.GridSearchCV(pipeline, params, cv = splits, scoring = 'f1')

    clf.fit(features_train, labels_train)
    gs = clf.best_estimator_

    dataset = data_pd.to_dict(orient="index")
#     dataset = data_pd_new.to_dict(orient="index")
    feature_list = ['poi'] + features_list[:17]
    test_classifier(gs, dataset, feature_list)

Gaussian_compare()



'''

Pipeline(steps=[('select', SelectKBest(k=17, score_func=<function f_classif at 0x118f769b0>)), ('pca', PCA(copy=True, n_components=8, whiten=False)), ('clf', GaussianNB())])
	Accuracy: 0.84821	Precision: 0.45939	Recall: 0.35350	F1: 0.39955	F2: 0.37058
	Total predictions: 14000	True positives:  707	False positives:  832	False negatives: 1293	True negatives: 11168
'''


GNB = GaussianNB()

pca = PCA(n_components = 8)
pca_fit = pca.fit_transform(data_pd[features_list[:17]], data_pd['poi'])

# print pca_fit.shape

pca_fit_dt = pd.DataFrame(pca_fit, index = data_pd.index, columns = features_list[:8])
pca_fit_dt.insert(0, 'poi', data_pd['poi'])

# print pca_fit_dt
dataset = pca_fit_dt.to_dict(orient = "index")

feature_list = ['poi'] + features_list[:8]
# print dataset
test_classifier(GNB, dataset, feature_list)


'''
GaussianNB()
	Accuracy: 0.85343	Precision: 0.48192	Recall: 0.34650	F1: 0.40314	F2: 0.36713
	Total predictions: 14000	True positives:  693	False positives:  745	False negatives: 1307	True negatives: 11255
'''


def SVC_compare():

    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('select',SelectKBest()),
                        ('pca', PCA()),
                         ('svm', SVC())
                    ])

    params = {
            'select__k' : range(10,17),
#               'pca__n_components' : range(3,10),
              'svm__kernel': ['rbf'],
              'svm__C': [0.1,1,10,100],
              'svm__gamma': [0.1,1,10,100]
    }

    clf = grid_search.GridSearchCV(pipeline, params, cv = splits, scoring = 'f1')
    clf.fit(features_train, labels_train)
    gs = clf.best_estimator_

#     clf = Pipeline([('select', SelectKBest(k = 10)),
#                    ('svm', SVC(kernel = 'rbf', C = 1))
#                    ])

#     data_pd = data_set('mnmx') # Doesn't matter even if I changed the data scale to log, sqrt, minmax, none.
    dataset = data_pd.to_dict(orient="index")
    feature_list = ['poi'] + features_list

    test_classifier(gs, dataset, feature_list)


SVC_compare()

'''
Pipeline(steps=[('select', SelectKBest(k=17, score_func=<function f_classif at 0x118f769b0>)), ('pca', PCA(copy=True, n_components=8, whiten=False)), ('clf', GaussianNB())])
	Accuracy: 0.84821	Precision: 0.45939	Recall: 0.35350	F1: 0.39955	F2: 0.37058
	Total predictions: 14000	True positives:  707	False positives:  832	False negatives: 1293	True negatives: 11168
'''

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(C=100, gamma=1, kernel='rbf'))
])

feature_list = ['poi'] + features_list[:16]
dataset = data_pd[feature_list].to_dict(orient="index")

test_classifier(pipeline, dataset, feature_list)

'''
Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
	Accuracy: 0.84350	Precision: 0.35682	Recall: 0.11900	F1: 0.17848	F2: 0.13730
	Total predictions: 14000	True positives:  238	False positives:  429	False negatives: 1762	True negatives: 11571
'''



def tree_compare():

    pipeline = Pipeline([
#                         ('scaler',StandardScaler()),
                        ('select',SelectKBest()),
                        ('pca', PCA()),
                        ('tree', DecisionTreeClassifier())
                        ])

    params = {
            'select__k' : [14],
              'pca__n_components' : [12],
              'tree__max_depth' : [8],
              'tree__min_samples_split' : [1]
             }

#     clf = grid_search.GridSearchCV(pipeline, params, scoring = 'recall')

#     print clf.get_params()
    clf = grid_search.GridSearchCV(pipeline, params, cv = splits, scoring = 'f1')
    clf.fit(features_train, labels_train)
    gs = clf.best_estimator_


#     data_pd = data_set('mnmx')
    dataset = data_pd.to_dict(orient="index")

    feature_list = ['poi'] + features_list

    test_classifier(gs, dataset, feature_list)

tree_compare()

'''
Pipeline(steps=[('select', SelectKBest(k=10, score_func=<function f_classif at 0x118f769b0>)), ('pca', PCA(copy=True, n_components=None, whiten=False)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'))])
	Accuracy: 0.80164	Precision: 0.30700	Recall: 0.30900	F1: 0.30800	F2: 0.30860
	Total predictions: 14000	True positives:  618	False positives: 1395	False negatives: 1382	True negatives: 10605
'''

tree = DecisionTreeClassifier(min_samples_split = 1, max_depth = 10)

feature_list = ['poi'] + features_list[:10]
pca = PCA()

tree_fit = pca.fit_transform(data_pd[features_list[:10]], data_pd['poi'])


tree_fit_dt = pd.DataFrame(tree_fit, index = data_pd.index, columns = features_list[:10])
tree_fit_dt.insert(0, 'poi', data_pd['poi'])

dataset = tree_fit_dt[feature_list].to_dict(orient="index")


test_classifier(tree, dataset, feature_list)


'''
Pipeline(steps=[('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'))])
	Accuracy: 0.79829	Precision: 0.29065	Recall: 0.28600	F1: 0.28831	F2: 0.28692
	Total predictions: 14000	True positives:  572	False positives: 1396	False negatives: 1428	True negatives: 10604
'''


# dataset = data_pd.to_dict(orient="index")
# feature_list = ['poi'] + features_list
# test_classifier(clf, dataset, feature_list)

'''
FINAL DECISION BETWEEN THE CLASSIFIER USED SO FAR
GAUSSIANNB
SUPPORT VECTOR MACHINE
DECISIONTREECLASSIFIER


The best one was the gaussianNB result.
'''

GNB = GaussianNB()

pca = PCA(n_components = 8)
pca_fit = pca.fit_transform(data_pd[features_list[:17]], data_pd['poi'])
pca_fit_dt = pd.DataFrame(pca_fit, index = data_pd.index, columns = features_list[:8])
pca_fit_dt.insert(0, 'poi', data_pd['poi'])

dataset = pca_fit_dt.to_dict(orient = "index")

feature_list = ['poi'] + features_list[:8]

pickle.dump(GNB, open("../final_project/my_classifier.pkl","wb"))

pickle.dump(dataset, open("../final_project/my_dataset.pkl","wb"))

pickle.dump(feature_list, open("../final_project/my_feature_list.pkl","wb"))

