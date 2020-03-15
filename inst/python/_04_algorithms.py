# -*- coding: utf-8 -*-
"""
Created on 01 March 2019
@author: megan.woods

- performs regression for algorithms

It is called in
    - _05_main
    - app
"""
import _01_constants as constants
import _02_my_functions as mf

import os
import numpy as np
from sklearn import linear_model, neighbors, tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#from sklearn import svm
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.neural_network import MLPRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#os.chdir(constants.current_dir)

# MODELS__________________________________________________________________________________
models = {#"Bayesian Ridge": linear_model.BayesianRidge(),
         # "Decision Tree Regression": tree.DecisionTreeRegressor(),
          "SVM":  svm.SVC(kernel='rbf', probability=False, gamma='scale', class_weight='balanced'),
          "KNN": KNeighborsClassifier(n_neighbors=5),
           "NB": GaussianNB(),
          #"Kernel Ridge": KernelRidge(),
          #"Kriging": GaussianProcessRegressor(),          
          #"LARS": linear_model.Lars(),
          #"Lars Lasso": linear_model.LassoLars(), # Convergence issues.........................
          #"Lasso":linear_model.Lasso(alpha=0.1), # Convergence issues...................................
          #"Linear Regression":linear_model.LinearRegression(),
          #"MLP": MLPRegressor(),
          #"Nearest Neighbors":neighbors.KNeighborsRegressor(n_neighbors=5),
          #"Perceptron": linear_model.Perceptron(),
          #"Ridge Regression":linear_model.Ridge(),
          #"RidgeCV Regression":linear_model.RidgeCV(alphas=[0.1,1.0,10.0],cv=3)})
          #"SVM": svm.SVR(),
          #"SGD": linear_model.SGDRegressor(),
          }

alg_names = list(models.keys())
#algs_to_scale = ["Bayesian Ridge","Linear Regression","Ridge Regression", "SGD"]
algs_to_scale = ["SVM", "KNN", "NB"]
#algs_to_not_scale = ["Decision Tree Regression", "Nearest Neighbors"]
regression_num = 1 # to differentiate regression plots

class Metamodel():
    """ Class to create a (linear or decision tree) regression model. Can plot
        the predicted vs. actual plots, residual plots, as well as display
        statistical performanes of the model
    """
    def __init__(self, reg_type, data):
        self.data = data
        self.target = data.target
        self.name = reg_type
        self.model = models[reg_type]
        self.model.fit(data.X_train, data.y_train)
        self.pred_train = self.model.predict(data.X_train)
        self.pred_test = self.model.predict(data.X_test)
        #self.mse = mean_squared_error(data.y_test, self.pred_test)
        #self.rmse = np.sqrt(self.mse)
        #self.r2 = r2_score(data.y_test, self.pred_test)
        self.recall = recall_score(data.y_test, self.pred_test, average='weighted')
            

    def predict_y(self, point):
        self.predicted_point = self.model.predict(point)

class Algorithms_Results():
    def __init__(self, data):

        ##### Step 1 #####
        # feature reduction

        ##### Step 2 #####
        # run regression on dataset
        models = {}
        for i in alg_names:
            models[i] = Metamodel(i, data)

        ##### Step 3 #####
        # find RMSE and r2 performances for each algorithm
        algorithm_results = [models[i] for i in models]

        #performances_rmse = {}
        #performances_r2 = {}
        performances_recall = {}

        for i in algorithm_results:
            performances_recall[i.name] = i.recall
            #performances_rmse[i.name] = round(i.rmse, 5)
            #performances_r2[i.name] = round(i.r2, 5)



        ##### Step 4 #####
        # rank the algorithms based on their RMSE performances
        #ranks = mf.find_ranks(performances_rmse)
        #ranks_ordered = mf.find_ranks(performances_rmse, return_sorted = True)

        # rank the algorithms based on their recall performances
        ranks = mf.find_ranks(performances_recall)
        ranks_ordered = mf.find_ranks(performances_recall, return_sorted=True)

        self.models = models
        self.name = data.name
        self.target = data.target 
        self.num_cols = data.num_pred_cols
        self.num_train = data.num_train
        self.num_test = data.num_test
        #self.performances_rmse = performances_rmse
        self.performances_recall = performances_recall
        #self.performances_r2 = performances_r2
        self.ranks = ranks
        self.ranks_ordered = ranks_ordered



