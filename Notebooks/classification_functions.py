import pandas as pd
import numpy as np
from statistics import mean

#modeling
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, auc, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

def x_GBoost(X_train, y_train):
    '''
    A fucntion for cross validating xgboost model

    Parameters
    ----------
    train data

    Returns
    -------
    prints cross validation classification metrics 
    returns xgboost model 
    '''
    #this helps with the way kf will generate indices below
    X, y = np.array(X_train), np.array(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=23) #randomly shuffle before splitting
    precision, recall, f1,  auc, logl, ac = [] , [], [], [], [], []
    for train_ind, val_ind in kf.split(X, y):
        
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        #xgboost params
        params = { 
                    'n_estimators':30000,
                    'max_depth':10,
                    'learning_rate':.3, 
                    'subsample':.8,
                    'min_child_weight':3,
                    'colsample_bytree':.8,
                    'random_state' : 0,
                    'verbosity' : 0,
                    'n_jobs' : -1}

        gbm = XGBClassifier()
        gbm.set_params(**params)
        gbm.fit(X_train, y_train)

        #calculate metrics and append to metric list
        metrics = calc_scores(gbm, X_val, y_val)
        ac.append(metrics[0])
        precision.append(metrics[1])
        recall.append(metrics[2])
        f1.append(metrics[3])
        auc.append(metrics[4])
        logl.append(metrics[5])

    #find mean of list and print scores
    ac = mean(ac)
    precision = mean(precision)
    recall = mean(recall)
    f1 = mean(f1)
    auc = mean(auc)
    logl = mean(logl)
    print(f'XGBoost:\n')
    get_scores(ac, precision, recall, f1, auc, logl)
          
    return gbm



def random_forest(X_train, y_train):
    '''
    A fucntion for fitting random forest model
    performs randomized cross validation search to find optimal hyperparameters

    Parameters
    ----------
    train data

    Returns
    -------
    prints cross validation classification metrics and params 
    returns random forest model 
    '''
    rf = RandomForestClassifier()
    #params to search
    rand_param = {
                    'n_estimators': [200, 500, 1000],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth' : [ 5, 10, 30],
                    'min_samples_leaf': [1, 4],
                    'min_samples_split': [2, 8]
                }
    rs = RandomizedSearchCV(rf, param_distributions= rand_param, cv=5, n_iter=10, n_jobs=-1)
    rs.fit(X_train, y_train)

    #get metrics and print metrics and params
    metrics = calc_cv_scores(rs, X_train, y_train)
    ac = metrics[0]
    precision = metrics[1]
    recall = metrics[2] 
    f1 = metrics[3]
    auc = metrics[4]
    logl = metrics[5]

    print(f'Random Forest with params:\n')
    print(rs.best_params_)
    get_scores(ac, precision, recall, f1, auc, logl)
          
    return rs

def decision_tree(X_train, y_train):
    '''
    A fucntion for fitting decision tree model
    performs randomized cross validation search to find optimal hyperparameters

    Parameters
    ----------
    train data

    Returns
    -------
    prints cross validation classification metrics and params 
    returns decsion tree model 
    '''
    dt = DecisionTreeClassifier()
    #params to search
    rand_params = {
                    'max_depth': [3,6,12, 20],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['auto', 'sqrt', 'log2'],
                }
    rs = RandomizedSearchCV(dt, param_distributions= rand_params, cv=5, n_iter=20, n_jobs=-1)
    rs.fit(X_train, y_train)

    #get metrics and print metrics and params
    metrics = calc_cv_scores(rs, X_train, y_train)
    ac = metrics[0]
    precision = metrics[1]
    recall = metrics[2] 
    f1 = metrics[3]
    auc = metrics[4]
    logl = metrics[5]

    print(f'Decision Tree with params:\n')
    print(rs.best_params_)
    get_scores(ac, precision, recall, f1, auc, logl)
          
    return rs
    

def logistic_model_scaled(X_train, y_train):
    '''
    A fucntion for fitting logistic regression model on scaled trainign data
    performs randomized cross validation search to find optimal hyperparameters

    Parameters
    ----------
    train data

    Returns
    -------
    prints cross validation classification metrics and params
    returns logistic model 
    '''
    scaler = StandardScaler() #scale data
    X_train_scaled = scaler.fit_transform(X_train)
    lm = LogisticRegression()

    #params to search
    rand_params = {
                    'max_iter': [10000],
                    'C': [0.1, 1, 10, 50],
                    'penalty': ['l1', 'l2']
                }
    rs = RandomizedSearchCV(lm, param_distributions= rand_params, cv=5, n_iter=10, n_jobs=-1)
    rs.fit(X_train_scaled, y_train)

    #get metrics and print metrics and params
    metrics = calc_cv_scores(rs, X_train_scaled, y_train)
    ac = metrics[0]
    precision = metrics[1]
    recall = metrics[2] 
    f1 = metrics[3]
    auc = metrics[4]
    logl = metrics[5]

    print(f'Logistic Regression with params:\n')
    print(rs.best_params_)
    get_scores(ac, precision, recall, f1, auc, logl)
          
    return rs

def knn_classification_scaled(X_train, y_train):
    '''
    A fucntion for fitting knn classification model on scaled trainign data
    performs randomized cross validation search to find optimal hyperparameters

    Parameters
    ----------
    train data

    Returns
    -------
    prints cross validation classification metrics and params
    returns knn model 
    '''
    scaler = StandardScaler() #scale data
    X_train_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsClassifier()

    #params to search
    rand_param = {
                    'n_neighbors': [3, 5, 7 , 10  ]   
                }
    rs = RandomizedSearchCV(knn, param_distributions= rand_param, cv=5, n_iter=3, n_jobs=-1)
    rs.fit(X_train_scaled, y_train)
    
    #calculate and print metrics and params
    metrics = calc_cv_scores(rs, X_train_scaled, y_train)
    ac = metrics[0]
    precision = metrics[1]
    recall = metrics[2] 
    f1 = metrics[3]
    auc = metrics[4]
    logl = metrics[5]

    print(f'KNN with params:\n')
    print(rs.best_params_)
    get_scores(ac, precision, recall, f1, auc, logl)
          
    return rs

def calc_scores(model, X_val, y_val):
    '''
    A fucntion for calculating calculation metrics 

    Parameters
    ----------
    model and test data

    Returns
    -------
    list of classification metrics
    '''
    preds = model.predict(X_val)
    y_val_enc = pd.get_dummies(y_val)
    probs = model.predict_proba(X_val)
    
    ac = round(model.score( X_val, y_val), 3)
    precision = (round(precision_score( y_val, preds, average='macro'), 3))
    recall = (round(recall_score( y_val, preds, average='macro'), 3))
    f1 = (round(f1_score( y_val, preds, average='macro'), 3))
    auc = (round(roc_auc_score( y_val_enc, probs, average='macro', multi_class='ovr'), 3))
    logl = (round(log_loss( y_val, probs), 3))

    return [ac, precision, recall, f1, auc, logl]

def calc_cv_scores(model, X_test, y_test):
    '''
    A fucntion for calculating calculation metrics using cross validation

    Parameters
    ----------
    model and test data

    Returns
    -------
    list of classification metrics
    '''
    ac = round(cross_val_score(model, X_test, y_test, scoring='accuracy', cv=5).mean(), 3)
    precision = round(cross_val_score(model, X_test, y_test, scoring='precision_macro', cv=5).mean(), 3)
    recall = round(cross_val_score(model, X_test, y_test, scoring='recall_macro', cv=5).mean(), 3)
    f1 = round(cross_val_score(model, X_test, y_test, scoring='f1_macro', cv=5).mean(), 3)
    auc = round(cross_val_score(model, X_test, y_test, scoring='roc_auc_ovr', cv=5).mean(), 3)
    logl = round(cross_val_score(model, X_test, y_test, scoring='neg_log_loss', cv=5).mean(), 3)

    return [ac, precision, recall, f1, auc, logl]

def get_scores(ac, precision, recall, f1, auc, logl):
    '''
    A fucntion for printing classification metrics

    Parameters
    ----------
    classification metrics
    '''
    print(f'Accuracy: {ac},\n'
          f'Precision score: {precision},\n'
          f'Recall score: {recall},\n'
          f'f1 score: {f1},\n'
          f'ROC AUC score: {auc},\n'
          f'Negative Log-loss: {logl},\n')
    

def confusion_matrix_heatmap(model, X_test, y_test):
    '''
    A fucntion for plotting the confusion matrix

    Parameters
    ----------
    model : model to be tested
    y_test, X_test : test data

    Returns
    -------
    heatmap of confusion matrix
    '''
    preds = model.predict(X_test)
    conf = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='d',
               xticklabels=['Linux', 'MacOS', 'Windows'],
               yticklabels=['Linux', 'MacOS', 'Windows'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')

def plot_roc(y_test, X_test, model):
    '''
    A fucntion for plotting the roc curve and comuting the area under the curve for each class

    Parameters
    ----------
    model : model to be tested
    y_test, X_test : test data

    Returns
    -------
    line graph of 3 roc curves
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #encode y_test and get probablities using model
    y_test_enc = pd.get_dummies(y_test)
    probs = model.predict_proba(X_test)

    #loop through all classes
    for i in range(3):
        #find roc values and area under the curve given class
        fpr[i], tpr[i], _ = roc_curve(y_test_enc.iloc[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #plot
    plt.figure()
    op_sys = y_test_enc.columns
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color, os in zip(range(3), colors, op_sys):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of {0} (area = {1:0.2f})'
                ''.format(os, roc_auc[i]))

    plt.plot([0, 1], [0, 1], '--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()





    