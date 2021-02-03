import pandas as pd
import numpy as np

#modeling
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, fbeta_score, auc, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_model(X_train, y_train, regularization, threshold, threshold_val, b):
    #this helps with the way kf will generate indices below
    X, y = np.array(X_train), np.array(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=23) #randomly shuffle before splitting
    precision, recall, f1, fbeta, auc, logl = [] , [], [], [], [], []

    for train_ind, val_ind in kf.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        lm = LogisticRegression(C=regularization, max_iter=10000)
        lm.fit(X_train, y_train)

        if threshold:
            preds = (lm.predict_proba(X_val)[:, 1] >= threshold_val)
        else:
            preds = lm.predict(X_val)
    
        logl.append(round(log_loss( y_val, preds), 3))
        precision.append(round(precision_score( y_val, preds, average='macro'), 3))
        recall.append(round(recall_score( y_val, preds, average='macro'), 3))
        f1.append(round(f1_score( y_val, preds, average='macro'), 3))
        fbeta.append(round(fbeta_score( y_val, preds, beta = b, average='macro'), 3)) #beta times more impotance to recall than precision
        
        y_val_enc = pd.get_dummies(y_val)
        preds_enc = pd.get_dummies(preds)
        auc.append(round(roc_auc_score( y_val_enc, preds_enc, average='macro', multi_class='ovr'), 3))

    print(f'logistic regression with C = {regularization}:\n'
          f'Precision score: {np.mean(precision)},\n'
          f'Recall score: {np.mean(recall)},\n'
          f'f1 score: {np.mean(f1)},\n'
          f'fbeta score for beta = {b}: {np.mean(fbeta)},\n'
          f'ROC AUC score: {np.mean(auc)},\n'
          f'Log-loss: {np.mean(logl)},\n')
    plot_roc(y_val, preds)
          
    return lm

def knn_classification(X_train, y_train, k, b):
    #this helps with the way kf will generate indices below
    X, y = np.array(X_train), np.array(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=23) #randomly shuffle before splitting
    precision, recall, f1, fbeta, auc, logl = [] , [], [], [], [], []

    knn = KNeighborsClassifier(n_neighbors=k) #k nearest neighbors
    knn.fit(X_train, y_train)

    for train_ind, val_ind in kf.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        precision.append(round(precision_score( y_val, knn.predict(X_val), average='macro'), 3))
        recall.append(round(recall_score( y_val, knn.predict(X_val), average='macro'), 3))
        f1.append(round(f1_score( y_val, knn.predict(X_val), average='macro'), 3))
        fbeta.append(round(fbeta_score( y_val, knn.predict(X_val), beta = b, average='macro'), 3)) #beta times more impotance to recall than precision
        logl.append(round(log_loss( y_val, knn.predict(X_val)), 3))

        y_val_enc = pd.get_dummies(y_val)
        preds_enc = pd.get_dummies(knn.predict(X_val))
        auc.append(round(roc_auc_score( y_val_enc, preds_enc, average='macro', multi_class='ovr'), 3))

    print(f'KNN Classification with k = {k}:\n'
          f'Precision score: {np.mean(precision)},\n'
          f'Recall score: {np.mean(recall)},\n'
          f'f1 score: {np.mean(f1)},\n'
          f'fbeta score for beta = {b}: {np.mean(fbeta)},\n'
          f'ROC AUC score: {np.mean(auc)},\n'
          f'Log-loss: {np.mean(logl)},\n')
    plot_roc(y_val, knn.predict(X_val))
    
    return knn

def conf_matrix(y_test, preds):

    conf = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='d',
               xticklabels=['Linux', 'MacOS', 'Windows'],
               yticklabels=['Linux', 'MacOS', 'Windows'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')

def plot_roc(y_test, preds):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_enc = pd.get_dummies(y_test)
    preds_enc = pd.get_dummies(preds)

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_enc.iloc[:, i], preds_enc.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(3):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    # fpr, tpr, thresholds = roc_curve(y_test, preds)

    # plt.plot(fpr, tpr,lw=2)
    # plt.plot([0,1],[0,1],c='violet',ls='--')
    # plt.xlim([-0.05,1.05])
    # plt.ylim([-0.05,1.05])

    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve for fraud problem')
    