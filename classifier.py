#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # dont show plots, just save

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from pandas.api.types import is_numeric_dtype
import pickle

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import multilabel_confusion_matrix
import pickle
import shutil
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Set the parameters of each model by cross-validation gridsearch
#from custom-perceptron import my_perceptron
models = {'KNN': KNeighborsClassifier(),
        'SVC': OneVsRestClassifier(SVC()),
        'LR': OneVsRestClassifier(LogisticRegression()),
        #   'GNB': OneVsRestClassifier(GaussianNB()),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(n_jobs=-1), 
        'MLP': MLPClassifier(learning_rate='adaptive',shuffle=True, max_iter=500),
        # 'LGBM': lgb.LGBMClassifier(),
        'XGB': XGBClassifier()}

param_grid = [{'KNN__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'KNN__weights': ['uniform', 'distance']},
            {'SVC__estimator__kernel': ['rbf', 'linear'], 'SVC__estimator__gamma': ['scale', 'auto'], 'SVC__estimator__C': [10, 100, 1000]},
            {'LR__estimator__penalty': ['l1', 'l2'], 'LR__estimator__C': [1.0, 0.5, 0.1], 'LR__estimator__solver': ['lbfgs','liblinear']},
            #   {'GNB__priors':[None, [0.5,0.5], [0.1, 0.9], [0.000001,0.99999], [0.000000001,0.99999999]]},
            {'DT__criterion': ['entropy'], 'DT__max_depth': [6], 'DT__min_samples_leaf': [1], 'DT__min_samples_split': [4]},
            {'RF__n_estimators': [200, 600], 'RF__max_depth': [4, 10, None], 'RF__min_samples_leaf': [1, 2, 5]},
            {'MLP__hidden_layer_sizes': [150,100,50], 'MLP__activation':['relu','logistic','tanh'], 
            'MLP__solver': ['adam','lbfgs','sgd']}, #,verbose=True, early_stopping=True
            # {'LGBM__objective': ['binary','multiclass','regression'],'LGBM__boosting_type': ['gbdt','dart','rf'],'LGBM__num_leaves': [31],
            #  'LGBM__learning_rate': [.1,.2,.3],'LGBM__feature_fraction': [0.9]},
            {'XGB__learning_rate': [.1,.2,.3], 'XGB__max_depth': [1, 2, 3, 4, 5, 6], 'XGB__min_child_weight': [1,2],
            'XGB__subsample': [1.0, 0.5, 0.1], 'XGB__n_estimators': [200, 600]}
             ]

feature_cols = ['Building Total Area','Reference area','Above-ground floors',
                'Underground floor','Energy consumption before',
                'Initial energy class ','Energy class after']

target_cols = ['Carrying out construction works ','Reconstruction of engineering systems',
                'Heat installation','Water heating system']

categorical_cols = ['Above-ground floors','Underground floor',
                    'Carrying out construction works ',
                    'Reconstruction of engineering systems',
                    'Heat installation','Water heating system',
                    'Initial energy class ','Energy class after']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def data_scaling(dframe, onehot_fields, categorical_scalers=None, train_scalers=None, train=False, target=False):

    # if target dataframe, then its a single-column df, a series. Need to convert it back to df
    if(target): dframe = pd.DataFrame(dframe)

    # Categorical variables: label encoding
    # Initialize a dictionary to store the scalers    
    # if not training set, then use existing scalers
    if(train):
        categorical_scalers = {column: LabelEncoder() for column in onehot_fields}
        # Scale each column and store in the dictionary
        for column, scaler in categorical_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.fit_transform(dframe[column])
    else:
        for column, scaler in categorical_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.transform(dframe[column])

    # Continuous variables: scaling
    continuous_fields = [col for col in dframe.columns if col not in onehot_fields]

    # Initialize a dictionary to store the scalers      
    # if not training set, then use existing scalers
    if(train):
        train_scalers = {column: MinMaxScaler() for column in continuous_fields}
        # Scale each column and store in the dictionary
        for column, scaler in train_scalers.items():            
            if(column in dframe.columns):
                dframe[column] = scaler.fit_transform(dframe[[column]])
    else:
        for column, scaler in train_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.transform(dframe[[column]])


    if(train):
        return dframe, train_scalers, categorical_scalers
    else:
        return dframe


def train_test_valid_split(dframe,stratify_cols='Region',target_cols='Electricity produced by solar panels'):
    """
    we choose to split data with validation/test data to be at the end of time series
    Parameters:
        dframe: pandas.dataframe containing dataframe to split
        stratify_cols: list of dframe cols to be used for stratification 
        target_cols: list of dframe columns to be used as target for training 
    Returns:
        pandas.dataframe containing train/test/valiation data
        pandas.dataframe containing valiation data
        pandas.dataframe containing test data
        1x2 sklearn scalers vector for X and Y respectively 
    """

    scalers = {}

    continuous_fields = [col for col in dframe.columns if col not in stratify_cols]

    # Remove NaNs / duplicates / outliers
    dframe = dframe.dropna().reset_index(drop=True)
    dframe.drop_duplicates(inplace=True)
    dframe = dframe[(np.abs(zscore(dframe[continuous_fields])) <= 3).all(axis=1)]

    strat_df = dframe[categorical_cols].copy()

    y = dframe[target_cols].copy()

    strat_df = dframe[stratify_cols].copy()

    dframe.drop(target_cols,axis=1,inplace=True)

    train_X, test_X, train_Y, test_Y = train_test_split(dframe, y, test_size=0.2, random_state=1, 
                                                        shuffle=True) # , stratify=strat_df

    # train right now is both train and validation set
    train_X, scalers['X_continuous_scalers'], scalers['X_categorical_scalers'] = data_scaling(train_X, stratify_cols, train=True)
    # train_Y, scalers['Y_continuous_scalers'], scalers['Y_categorical_scalers'] = data_scaling(train_Y, stratify_cols, train=True, target=True)

    test_X = data_scaling(test_X, stratify_cols, scalers['X_categorical_scalers'], scalers['X_continuous_scalers'])
    # test_Y = data_scaling(test_Y, stratify_cols, scalers['Y_categorical_scalers'], scalers['Y_continuous_scalers'], target=True)

    # strat_df = pd.concat([strat_df.pop(x) for x in stratify_cols], axis=1)

    # print(train_X); print(stratify_cols)
    train_stratify_cols = [item for item in train_X.columns if item in categorical_cols]
    strat_df = train_X[train_stratify_cols].copy()
    # print(strat_df.info())

    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=None, random_state=42, 
                                                      shuffle=True) #, stratify=strat_df # 0.25 x 0.8 = 0.2
    
    # train_X = train_X[feature_cols].copy(); train_Y = train_Y[target_cols].copy()
    # print(train_Y.head(5))
    with open('./models-scalers/service_1_scalers.pkl', 'wb') as f: pickle.dump(scalers, f)

    return train_X, validation_X, test_X, train_Y, validation_Y, test_Y, scalers

def modelSearch(train_X, train_Y, test_X, test_Y):
  """
  This function is used to implement hyperparameter search 

  """
  best_scores=[]
  params=[]

  for (classifier, model_params, name) in list(zip(models.values(), param_grid, models.keys())):
      print(f"\nTuning hyper-parameters, based on accuracy for {name} with parameter grid: {model_params}")

      pipe = Pipeline([(name, models[name])])
      clf = GridSearchCV(estimator=pipe, param_grid=model_params, cv=5, scoring='accuracy', n_jobs=-1)    
      clf.fit(train_X, train_Y) 

      # print(f"Mean performance of each parameter combination based on Cross Validation")
      # performance = pd.DataFrame(clf.cv_results_['params'])
      # performance["Score"] = clf.cv_results_['mean_test_score']
      # print(performance)

      # print("\nBest parameters set found on training set:")
      # print(clf.best_params_)
      # params.append(clf.best_params_)

      # print("\nThe scores are computed on the full evaluation set:")
      #evaluate and store scores of estimators of each category on validation set
      score = clf.score(test_X.values, test_Y.values)
      print("Accuracy:", score)
      best_scores.append(score)

      with open(f'./models-scalers/{name}.pkl', 'wb+') as f: pickle.dump(clf, f)
      
      # pred_Y = clf.predict(test_X)
      # print(metrics.classification_report(test_Y, pred_Y, digits=5))
      # confusion_matrices = multilabel_confusion_matrix(test_Y, pred_Y)
      # for i, cm in enumerate(confusion_matrices):
      #     print(f"Confusion Matrix for Class \"{target_cols[i]}\":\n {cm}\n")
          
      #     # ax, labels, title and ticks
      #     ax= plt.subplot();
      #     sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens);  #annot=True to annotate cells, ftm='g' to disable scientific notation
      #     ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');  
      #     ax.set_xticklabels(['No','Yes']); ax.set_yticklabels(['No','Yes'])
      #     print(f"True Positives: {cm[1,1]}, False Positives: {cm[0,1]}, True Negatives: {cm[0,0]}, False Negatives: {cm[1,0]} \n\n")
      #     plt.title(target_cols[i]); plt.show()
  return best_scores

###############################################################################
# Evaluation with various classification metrics (classification report)
###############################################################################

def model_evalutate(best_model, best_model_index, best_scores, test_X, test_Y):

    with open(f'./models-scalers/{best_model}.pkl', 'rb') as f: clf = pickle.load(f)

    # pipe = Pipeline([(best_model, models[best_model])])
    # clf = GridSearchCV(estimator=pipe, param_grid=param_grid[best_model_index], cv=5, scoring='accuracy', n_jobs=-1)    
    # clf.fit(train_X, train_Y) 

    print(f"Mean performance of each parameter combination based on Cross Validation")
    performance = pd.DataFrame(clf.cv_results_['params'])
    performance["Score"] = clf.cv_results_['mean_test_score']
    print(performance)

    print("\nBest parameters set found on training set:")
    print(clf.best_params_)

    #evaluate and store scores of estimators of each category on validation set
    score = clf.score(test_X.values, test_Y.values)
    print("\nAccuracy:", score)
    best_scores.append(score)

    pred_Y = clf.predict(test_X.values)
    print(metrics.classification_report(test_Y, pred_Y, digits=5))
    confusion_matrices = multilabel_confusion_matrix(test_Y, pred_Y)
    for i, cm in enumerate(confusion_matrices):
        print(f"Confusion Matrix for Class \"{target_cols[i]}\":\n {cm}\n")

        # ax, labels, title and ticks
        ax= plt.subplot();
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');  
        ax.set_xticklabels(['No','Yes']); ax.set_yticklabels(['No','Yes'])
        plt.title(target_cols[i]); plt.show()
        print(f"True Positives: {cm[1,1]}, False Positives: {cm[0,1]}, True Negatives: {cm[0,0]}, False Negatives: {cm[1,0]} \n\n")
        plt.savefig(f'./plots/{target_cols[i]}.png', bbox_inches='tight')
        plt.close()

def service_1_model_predict(best_model,service_1_targets,dict):
    with open('./models-scalers/service_1_scalers.pkl', 'rb') as f: scalers = pickle.load(f)
    with open(f'./models-scalers/best_classifier.pkl', 'rb') as f: clf = pickle.load(f)
    
    specs = pd.DataFrame.from_dict(dict) 
    
    # print(specs.head())
    # print(specs.info())
    scaled_specs = data_scaling(specs, categorical_cols, scalers['X_categorical_scalers'], scalers['X_continuous_scalers'])

    pred_Y = clf.predict(scaled_specs).ravel().tolist()
    
    pred_dict = {service_1_targets[i]: pred_Y[i] for i in range(len(service_1_targets))}

    # print(pred_dict)

    return pred_dict

def remove_files_by_key_part(directory, final_scores):
    print(directory)
    print(final_scores)
    [print(os.path.join(directory, filename)) for filename in os.listdir(directory) for key in final_scores if key in filename]
    [os.remove(os.path.join(directory, filename)) for filename in os.listdir(directory) for key in final_scores if key in filename]

def forecasting_model():
    df = pd.read_csv('./datasets/EF_comp.csv')
    # print(df.info())

    df = df[df.columns[df.columns.isin(feature_cols+target_cols)]]
    # print(df.info())

    # df = temp_df.copy()
    train_X, validation_X, test_X, train_Y, validation_Y, test_Y, scalers = train_test_valid_split(df,categorical_cols,target_cols) 

    best_scores = modelSearch(train_X, train_Y, test_X, test_Y)

    keys = list(models.keys())
    final_scores = {keys[i]: best_scores[i] for i in range(len(keys))}
    # print(final_scores)

    best_model = max(key for key, value in final_scores.items() if value == max(final_scores.values()))

    #make a copy of the best model based on "best model"
    directory = './models-scalers'
    shutil.copy(f'{directory}/{best_model}.pkl',f"{directory}/best_classifier.pkl")

    # remove_files_by_key_part(directory, final_scores)    

    best_model_index = list(models).index(best_model)
    model_evalutate(best_model, best_model_index, best_scores, test_X, test_Y)

    # dict = [{'Building Total Area': 351.6, 
    #         'Reference area': 277.4, 
    #         'Above-ground floors': 3, 
    #         'Underground floor': 0,
    #         'Initial energy class ': 'D',
    #         'Energy consumption before': 106.04,
    #         'Energy class after': 'B'}]

    service_1_model_predict(best_model, target_cols, dict)


if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    forecasting_model()