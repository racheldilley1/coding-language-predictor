import pandas as pd

#modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_model(X_train, y_train, class_type):
    lm = LogisticRegression(solver='newton-cg', multi_class=class_type)
    lm.fit(X_train, y_train)
    print(lm.score(X_train, y_train))
    return lm

def knn_classification(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(knn.score(X_train, y_train))
    return knn

def conf_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    print(y_test.value_counts())
    conf = confusion_matrix(y_true = y_test, y_pred = preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf, cmap=plt.cm.get_cmap('Blues'), annot=True, square=True, fmt='d',
               xticklabels=['Linux', 'MacOS', 'Windows'],
               yticklabels=['Linux', 'MacOS', 'Windows'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')