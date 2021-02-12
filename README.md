# Operating System Predictor

Play around with the model and make predictions using the [Operating System Predictor](http://192.168.1.71:8501/) web app that I created using [Streamlit](https://www.streamlit.io/).

-----------------

### Objective:

Build a supervised classifiaction model to to predict whether a coder uses a Windows, MacOS, or Linux operating system. 

-----------------

### Approach:

The features used in the model were age, region, gender, education level, undergrad major, age user started coding, years of coding experience, years of professional coding experience, number of databases they are familiar with or have used, the top coding languages thay are familiar with or hav used, and what developer type(s) or occupation they are working in or have worked in. 

Baseline modeling was done using only a select few features (age, region, gender, undergrad major, years of coding experience, number of databases they are familiar with or have used, and the top coding languages thay are familiar with or have used). A randomized search cross validation technique was used for logistic regression (with scaled data), KNN classification (with scaled data), decision tree and random forest to find the optimal hyperparamters for each and hyperparemter selection was done manually for the gradient boosting model. An F1 (due to the fact that we want recall and precision weighed equally) score and confusion matrix was calcualted for each of the five baseline models. Gradient boosting performed the best and was chosen for the final model. Additionaslly, synthetic minority oversampling technique was used to fix class imbalance (a majority of the target data was Windows) and hyperparameter tuning was performed on gradient boosting in order to optimize for F1 score. 

-----------------

### Featured Techniques:

* PostgreSQL
* Supervised Machiene Learning
* Logistic Regression
* KNN Classification
* Decision Tree
* Random Forest
* Gradient Boosting
* Synthtic Monirity Oversampling (SMOTE)
* Streamlit

-----------------

### Data:

Over 70,000 records used to create the model were collected from 2019 and 2020 [Stack Overflow survey data](https://insights.stackoverflow.com/survey). A [county dataset](https://www.kaggle.com/fernandol/countries-of-the-world#__sid=js0) was used to get data for the region feature. 

-----------------

### Results Summary:

A gradient boosting model was chosen with hyperparameters of  estimators, max depth, learning rate, subsample, minimum child weight, and column sample by tree to optimize for F1. On the test data, the model had an F1 of and an ROC AUC of . 

