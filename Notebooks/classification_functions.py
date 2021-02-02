import pandas as pd
import numpy as np

#modeling
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_model(X_train, y_train, regularization, threshold, threshold_val):
    #this helps with the way kf will generate indices below
    X, y = np.array(X_train), np.array(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=23) #randomly shuffle before splitting
    precision, recall, log_score = [] , [], []

    for train_ind, val_ind in kf.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        lm = LogisticRegression(C=regularization)
        lm.fit(X_train, y_train)

        if threshold:
            preds = (lm.predict_proba(X_val)[:, 1] >= threshold_val)
        else:
            preds = lm.predict(X_val)

        log_score.append(round(lm.score( X_val, y_val), 3))
        precision.append(round(precision_score( y_val, preds), 3))
        recall.append(round(recall_score( y_val, preds), 3))

    print(f'logistic regression with C = {regularization}:\n'
          f'Logistic score: {np.mean(log_score)},\n'
          f'Precision score: {np.mean(precision)},\n'
          f'Recall score: {np.mean(recall)},\n')
    return lm

def knn_classification(X_train, y_train, k):
    #this helps with the way kf will generate indices below
    X, y = np.array(X_train), np.array(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=23) #randomly shuffle before splitting
    precision, recall = [] , []

    knn = KNeighborsClassifier(n_neighbors=k) #k nearest neighbors
    knn.fit(X_train, y_train)

    for train_ind, val_ind in kf.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        precision.append(round(precision_score( y_val, knn.predict(X_val)), 3))
        recall.append(round(recall_score( y_val, knn.predict(X_val)), 3))

    print(f'KNN Classification with k = {k}:\n'
          f'Precision score: {np.mean(precision)},\n'
          f'Recall score: {np.mean(recall)},\n')
    
    return knn

def conf_matrix(model, X_test, y_test, threshold, threshold_val):
    if threshold:
        preds = (model.predict_proba(X_test)[:, 1] >= threshold_val)
    else:
        preds = model.predict(X_test)
    
    conf = confusion_matrix(y_true = y_test, y_pred = preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='d',
               xticklabels=['Linux', 'MacOS', 'Windows'],
               yticklabels=['Linux', 'MacOS', 'Windows'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')