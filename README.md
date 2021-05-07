# fintech-machine-learning-homework

This repository contains the Jupyter notebook, accompanied with imbalanced-learn and Scikit-learn libraries as part of the Fintech homework assignment Unit 11 - Risky Business.

In this homework assignment, we are helping fictitous investors to predict credit risk. We are using several machine learning techniques such as resampling and ensemble learning to train and evaluate our models with imbalanced classes.


## Files

### [Resampling](credit_risk_resampling.ipynb)

For this section, these are the following tasks:

- Read and convert the data into interger.

- Split the data into training and testing sets by separating feature and target datasets.

- Pre-process the data by using `StandardScaler` from `sklearn.preprocessing`. As we know, StandardScaler will transform the data such that its distribution will have a mean value 0 and standard deviation of 1. In case of multivariate data, this is done feature-wise.

- Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

- Build and evaluate the performance of several machine learning models to predict the credit risk of each model by comparing `balanced accuracy score`, `confusion matrix` and `imbalanced classification report` scores. The model we build by using following machine learning techniques:

    * Simple Logistic Regression
    
    * Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms
    
    * Undersample the data using the `Cluster Centroids` algorithm
    
    * Over- and undersample using a combination `SMOTEENN` algorithm
    


### [Ensemble Learning](credit_risk_ensemble.ipynb)

In this section, we train and compare two different ensemble classifiers to predict loan risk and evaluate each model. These are the following tasks:

- Read and convert the data into interger by using `get_dummies` and `labelencoder` function.

- Split the data into training and testing sets by separating feature and target datasets.

- Pre-process the data by using `StandardScaler` from `sklearn.preprocessing`.

- Build and evaluate the performance of several ensemble classifiers to predict the credit risk of each model by comparing `balanced accuracy score`, `confusion matrix` and `imbalanced classification report` scores. The model we build by using following ensemble algorithms: 

     * Balanced Random Forest Classifier
     
     * Easy Ensemble Classifier

- For the balanced random forest classifier, we sorted the important features in descending order along with the feature score.