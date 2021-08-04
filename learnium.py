import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

st.title('LEARNIUM- Streamlit WebApp for learning ML Classifiers')

st.write("""
Explore different classifier with 2 datasets, Iris and Wine
# KNN- Simplest ML algo. based on Supervised Learning technique.
# SVM- Popular Supervised Learning algo, which is used for Classification and Regression problems.
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Your Dataset',
    ('Iris', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select Your classifier',
    ('KNN', 'SVM')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else: 
        clf_name == 'KNN'
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else: 
        clf_name == 'KNN'
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='winter')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
