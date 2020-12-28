#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:24:03 2020

@author: yiannimercer
"""


#import libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
!pip install lightgbm
import lightgbm as lgb
import gc

##Minor cleanup

#Drop target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
#Feature Names
features = list(train.columns)

#Copy of testing data
test = app_test.copy()

#Median imputation of missing values 
imputer = SimpleImputer(strategy='median')

#Scale each feature 0-1
scaler = MinMaxScaler(feature_range=(0,1))

#Fit on the training data
imputer.fit(train)

#Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

#Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

##Logistic Regression

#Make model with the specified regularization parameter
logr = LogisticRegression(C = 0.0001)

logr.fit(train,train_labels)

logr_predictions = logr.predict_proba(test)[:,1]

#Logr Dataframe
logr_df = app_test[['SK_ID_CURR']]
logr_df['TARGET'] = logr_predictions

#Saving to csv file
logr_df.to_csv('LogisticRegression_Baseline_Predictions.csv',index = False)

##Random Forest  

rf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

#Train
rf.fit(train, train_labels)

#Extract Feature importance
feature_importance_values = rf.feature_importances_
feature_importances = pd.DataFrame({'feature':features, 'importance':feature_importance_values})

#Predict on test data
rf_predictions = rf.predict_proba(test)[:,1]

#Rf Dataframe
rf_df = app_test[['SK_ID_CURR']]
rf_df['TARGET'] = rf_predictions

#Saving to a csv file
rf_df.to_csv('RandomForest_Baseline_Predictions.csv',index = False)

##Support Vector Machine 

svm = SVC(C=1.0)

#Train
svm.fit(train,train_labels)

#Predict on test data
svm_predictions = svm.predict_proba(test)[:,1]

## Random Forest with Engineered Features

poly_features_names = list(app_train_poly.columns)

imputer = SimpleImputer(strategy='median')

poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

#Scale the polynomial features 
scaler = MinMaxScaler(feature_range=(0,1))

poly_features = scaler.fit_transform(poly_features)
    
poly_features_test = scaler.transform(poly_features_test)

rf_poly = RandomForestClassifier(n_estimators=100, random_state = 50, verbose = 1, n_jobs=-1)

#Train on the training data
rf_poly.fit(poly_features,train_labels)

#Make predictions on the test data
rf_poly_predictions = rf_poly.predict_proba(poly_features_test)[:,1]

#Rf Polynomial Df
rf_poly_df = app_test[['SK_ID_CURR']]
rf_poly_df['TARGET'] = rf_poly_predictions

#Saving to csv file 
rf_poly_df.to_csv('RandomForest_Baseline_Engineered_Features.csv',index = False)

##Random Forest with Domain Features

app_train_domain = app_train_domain.drop(columns = 'TARGET')

domain_features_names = list(app_train_domain.columns)

#Impute the domainnomial features
imputer = Imputer(strategy = 'median')

domain_features = imputer.fit_transform(app_train_domain)
domain_features_test = imputer.transform(app_test_domain)

#Scale the domainnomial features
scaler = MinMaxScaler(feature_range = (0, 1))

domain_features = scaler.fit_transform(domain_features)
domain_features_test = scaler.transform(domain_features_test)

random_forest_domain = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

#Train on the training data
random_forest_domain.fit(domain_features, train_labels)

#Extract feature importances
feature_importance_values_domain = random_forest_domain.feature_importances_
feature_importances_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})

#Make predictions on the test data
predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]

#Rf Domain Df
rf_domain_df = app_test[['SK_ID_CURR']]
rf_domain_df['TARGET'] = predictions

#Save to csv file
rf_domain_df.to_csv('RandomForest_Domain_Baseline.csv',index = False)


##Light Gradient Boosting Model with Original Features


def model(features, test_features, encoding = 'ohe', n_folds = 5):

    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] =  label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics

submission, fi, metrics = model(app_train, app_test)
print("Baseline Metrics")
print(metrics)

submission.to_csv('LightGB_Baseline.csv',index = False)

## Light Gradient Boosting Model with Domain Features 
app_train_domain['TARGET'] = train_labels

#Test domain knowledge features 
submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)

submission_domain.to_csv('baseline_lgb_domain_features.csv', index = False)









    