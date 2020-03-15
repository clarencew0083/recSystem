# -*- coding: utf-8 -*-
"""
@author: megan.woods
modified by Clarence Williams

It is called in
    - app
"""
# import packages_________________________________________________________________________
# ________________________________________________________________________________________
import _01_constants as constants
import _02_my_functions as mf
import _03_prepare_data as prep
import _04_algorithms as algs

import os
import pandas as pd
import numpy as np
from operator import sub, truediv
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
#from pydataset import data as pydata
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


#os.chdir(constants.current_dir)


# SETUP___________________________________________________________________________________
# ________________________________________________________________________________________

def get_meta_features(my_list):
    """ Creates dataframe of datasets and their metafeature values
    """
    return pd.concat([i.meta_features for i in my_list])


# def alg_rankings(my_list):
#    """ Creates dataframe of datasets and their rankings for each meta-model
#    """
#    data = [algs.Algorithms_Results(my_list[i]).ranks.values() for i in range(len(my_list))]
#    alg_results = pd.DataFrame(data=data, columns=algs.models.keys(), index=dataset_names)
#    return alg_results

def recommend(my_target_dataset, datasets, meta_features, recall_values):
    global num_correct
    global run
    global num_rel_perf_equal_0

    # Target dataset
    #print(my_target_dataset, flush=True)
    target_dataset = my_target_dataset # set the target dataset

    target_meta_features = [meta_features.loc[target_dataset].values]
    target_data = meta_features.loc[[target_dataset]] # target's meta_features
    #target_actual_rmses = rmse_values.loc[target_dataset]
    #target_actual_recall = recall_values.loc[target_dataset]
    #target_normalize_value = range_list[my_target_dataset]
    #target_actual_nrmses = target_actual_rmses / target_normalize_value
    # training datasets
    sc = StandardScaler()
    meta_X_train = meta_features.drop(target_dataset, 0)
    meta_X_train = sc.fit_transform(meta_X_train)
    #meta_y_train = rmse_values.drop(target_dataset, 0)
    # meta_y_train = recall_values.drop(target_dataset, 0)
    meta_y_train = recall_values

    # Build Linear Regression model
    #model = LinearRegression()

    #model = SVR(C=1, epsilon=0.1, gamma='scale')
    #model.fit(meta_X_train, meta_y_train)

    svm_pred = meta_model(meta_X_train, meta_y_train, target_data, 'SVM')
    knn_pred = meta_model(meta_X_train, meta_y_train, target_meta_features, 'KNN')
    nb_pred = meta_model(meta_X_train, meta_y_train, target_meta_features, 'NB')
    #dt_pred = meta_model(meta_X_train, meta_y_train, target_meta_features, 'DT')
    #rf_pred = meta_model(meta_X_train, meta_y_train, target_meta_features, 'RF')


   # frames_pred = [svm_pred, knn_pred, nb_pred]
    frames_pred = [svm_pred[0], knn_pred[0], nb_pred[0]]
    #combined_pred = pd.concat(frames_pred, axis=1)

    # Make rmse predictions
    #target_predicted_rmses = model.predict(target_meta_features)[0].tolist()
    #target_predicted_nrmses = np.divide(target_predicted_rmses, target_normalize_value)
    #target_predicted_nrmses = target_predicted_rmses / target_normalize_value

    # Make recall predictions
    target_predicted_recall = frames_pred
    #target_predicted_recall = combined_pred[0].tolist()



    # Spearman's Rank Correlation
    #corr, p_value = spearmanr(target_actual_nrmses, abs(target_predicted_nrmses))
    #corr, p_value = spearmanr(target_actual_recall, target_predicted_recall)
    #spearman_correlation.append(corr)

    # NRMSEs
    #nrmse_zippedlist = list(zip(target_actual_nrmses, target_predicted_nrmses, abs(target_predicted_nrmses)))
    #nrmse_comparisons = pd.DataFrame(nrmse_zippedlist,
     #                                columns=["Actual NRMSE", "Predicted NRMSE", "Abs Predicted NRMSE"],
     #                                index=rmse_values.columns)

    recall_zippedlist = list(zip( target_predicted_recall))
    recall_comparisons = pd.DataFrame(recall_zippedlist,
                                     columns=["Predicted Recall"],
                                     index=recall_values.columns)


    # Results
    #actual_best = recall_comparisons["Actual Recall"].idxmax()
    #actual_best = recall_comparisons["Actual Recall"].argmax()
    predicted_best = recall_comparisons["Predicted Recall"].idxmax()
    #actual_best_recall = recall_comparisons["Actual Recall"].max()
    #predicted_best_actual_recall = recall_comparisons.loc[predicted_best, "Actual Recall"]


    #temp_relative_performance = actual_best_recall / predicted_best_actual_recall

    #if temp_relative_performance == 0:
    #    num_rel_perf_equal_0 += 1
    #    diff_rel_perf_equal_0.append(predicted_best_actual_recall - actual_best_recall)
    #    diff_rel_perf_equal_0_names.append(target_dataset)

    #if predicted_best_actual_recall == 0:
    #    relative_performance.append(1)
    #    subset_rel_perf.append(1)
    #else:
    #    relative_performance.append(temp_relative_performance)
    #    if temp_relative_performance != 0:
    #        subset_rel_perf.append(temp_relative_performance)


    #temp_relative_performance = actual_best_recall / predicted_best_actual_recall

    #relative_performance.append(temp_relative_performance)
    #if temp_relative_performance != 0:
    #    subset_rel_perf.append(temp_relative_performance)

    #    print(nrmse_comparisons)
    #print("\n")
    #print(target_dataset + " Actual best ", actual_best)
    #print("Predicted best: ", predicted_best, flush=True)
    #print("Actual best recall ", actual_best_recall)
    #print("Predicted best actual recall ", predicted_best_actual_recall)
    #print("\n")
    
    return predicted_best




def meta_model(combined_meta, metric_df, target_set, algorithm_name):

    idx = metric_df.columns.get_loc(algorithm_name)
    m, n = combined_meta.shape
    #pca = PCA(n_components=3)
    #y_pred = np.zeros(shape=(m, 1))
    #pca_train_data = pca.fit_transform(X_train)
    #pca_test_data = pca.transform(X_test)
    #y_train, y_test = metric_df.iloc[train_index, idx], metric_df.iloc[test_index, idx]
    idx = metric_df.columns.get_loc(algorithm_name)
    y_train = metric_df.iloc[:, idx]
    # Calculate actual gamma values to test
    model = SVR(C=1, epsilon=0.1, gamma='scale')

    # model = linear_model.LinearRegression()
    model.fit(combined_meta, y_train)

    y_pred = model.predict(target_set)


    #data = pd.DataFrame(y_pred, index=metric_df.index)
    #data.columns = [algorithm_name]

    return y_pred




def rmse_results(my_list):
    """ Creates dataframe of datasets and their rmse values for each meta-model
    """
    data = [algs.Algorithms_Results(my_list[i]).performances_rmse.values() for i in range(len(my_list))]
    rmse_results = pd.DataFrame(data=data, columns=algs.models.keys(), index=dataset_names)
    return rmse_results

def recall_results(my_list):
    """ Creates dataframe of datasets and their rmse values for each meta-model
    """
    data = [algs.Algorithms_Results(my_list[i]).performances_recall.values() for i in range(len(my_list))]
    recall_results = pd.DataFrame(data=data, columns=algs.models.keys(), index=dataset_names)
    return recall_results


def get_normalizer(my_list, datasets):
    """
    """
    my_max = [i.y_train.max() for i in datasets]
    my_min = [i.y_train.min() for i in datasets]

    range_list = list(map(sub, my_max, my_min))
    return range_list

def rec_sys(d5, targetCol, meta_features, recall):
    valid_datasets = []
    # CSV Datasets
    #d1 = ["top_gear", constants.data_dir, "top_gear_edited.csv", "Verdict"]
    #d1 = ["heart", constants.data_dir, 'heart.csv', "target"]
    #d2 = ["spam", constants.data_dir, 'spam7.csv', 'yesno_bin']
    #d3 = ["bank_personal_loan", constants.data_dir, 'Bank_Personal_Loan2.csv', 'Personal Loan']
    #d4 = ["framingham", constants.data_dir, 'framingham2.csv', 'TenYearCHD']
    #d6 = ['Credit_Card_Fraud', constants.data_dir, '699.csv', 'isFraud']
    #d7 = ['zoo', constants.data_dir, 'zoo.csv', 'Col 18']
    #d8 = ['Wine Quality Red', constants.data_dir, 'winequality-red.csv', 'quality']
    #d9 = ['Wine Quality White', constants.data_dir, 'winequality-white.csv', 'quality']
    #d10 = ['Wine', constants.data_dir, 'wine.csv', 'Col 1']
    #d11 = ['Wilt', constants.data_dir, 'wilt.csv', 'class']
    #d12 = ['Wiki', constants.data_dir, 'wiki.csv', 'BI1']
    #d13 = ['Wholesale customers', constants.data_dir, 'Wholesale_customers_data.csv', 'Region']
    #d14 = ['Student Eval', constants.data_dir, 'turkiye-student-evaluation_R_Specific.csv', 'Q11']
    #d15 = ['Transfusion', constants.data_dir, 'transfusion.csv', 'donated']
    #d16 = ['breast_cancer', constants.data_dir, 'breast_cancer.csv', 'class']
    #d17 = ['Caesarian', constants.data_dir, 'caesarian.csv', 'Caesarian']
    #d18 = ['Cervival Cancer', constants.data_dir, 'risk_factors_cervical_cancer.csv', 'Dx:Cancer']


    #potential_datasets_CSVs = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d14, d15, d16, d17, d18]
    #potential_datasets_CSVs = [d5]
    #potential_datasets_CSVs = [d1, d2, d3, d4, d5]
    #potential_datasets_CSVs = [d1, d11]
    #potential_datasets_CSVs = []


    # run CSV datasets through PrepData

    prepared_data = prep.Prep_Data(name='data',  dataframe=d5, target_col=targetCol)
    if prepared_data.remove == False:
        valid_datasets.append(prepared_data)

    #potential_datasets_packages = list(pydata()["dataset_id"])
    #potential_datasets_packages = mf.unique(potential_datasets_packages)

    # indx = list(range(0,200))
    # indx.append(555) # this dataset is huge
    # potential_datasets_packages = [potential_datasets_packages[i] for i in indx]
    # 555 taking forever

    # run package datasets through PrepData
    #for i in potential_datasets_packages[:26]:
        #if i == "mhtdata":
        #    continue
        #if i == "Bundesliga":  # huge...
        #    continue
        #print(i)
        #prepared_data = prep.Prep_Data(name=i)
        #if prepared_data.remove == False:
        #    print(i)
        #    valid_datasets.append(prepared_data)

    # set variables___________________________________________________________________________
    datasets = valid_datasets  # list of instances of Prep_Data
    dataset_names = [datasets[i].name for i in range(len(datasets))] 

    meta_features.set_index('Name', inplace=True)
    #meta_features.set_index('Name')# list of dataset names
    meta_features_df = get_meta_features(datasets)
    #meta_features = pd.read_csv("C:\\Users\\c3_wi\\Desktop\\Python\\Data\\" + 'meta_features2.csv', index_col=0)
    meta_features = meta_features.append(meta_features_df)
    # algorithm_rankings = alg_rankings(datasets)
    #rmse_values = rmse_results(datasets)
    #recall = pd.read_csv("C:\\Users\\c3_wi\\Desktop\\Python\\Data\\" + 'recall2.csv', index_col=0)
    recall.set_index('Name', inplace=True)
    recall_values = recall.loc[:, 'SVM':]
    

    #range_list = get_normalizer(datasets)
    #alg_class = [algs.Algorithms_Results(datasets[i]) for i in range(len(datasets))]

    #normalized_rmse_values = pd.DataFrame(index=rmse_values.index)
    #for i in rmse_values.columns:
    #    normalized_rmse_values[i] = list(map(truediv, rmse_values[i], range_list))

    #excel_data = pd.concat([meta_features, normalized_rmse_values], axis=1)
    #excel_data.to_csv(constants.current_dir + "rmse.csv")

    #excel_data = pd.concat([meta_features, recall_values], axis=1)
    #excel_data.to_csv(constants.current_dir + "recall.csv")

    spearman_correlation = []
    relative_performance = []  # higher is better
    temp_list = []
    subset_rel_perf = []

    num_rel_perf_equal_0 = 0
    diff_rel_perf_equal_0 = []
    diff_rel_perf_equal_0_names = []

    num_correct = 0




    
    DATASETS = len(datasets)
    ITERATIONS = 1  # no point in changing this...
    TOTAL_RUNS = DATASETS * ITERATIONS

    
    predicted_best = recommend('data', datasets, meta_features, recall_values)
            
    return prepared_data.cleaned_data, predicted_best
    #
    #
    #print("\n")
    #print("Accuracy: ", num_correct / TOTAL_RUNS)
    #print("Number Correct:", num_correct)
    #print("Runs:", TOTAL_RUNS)
    #print("Realtive performance: ", sum(relative_performance) / TOTAL_RUNS)
    #print("Spearman's Rank: ", sum(spearman_correlation) / TOTAL_RUNS)

    #print("\n")
    #print("Num actual == 0: ", num_rel_perf_equal_0)
    #print("Max diff for these: ", 0)
    #print("Min diff for these: ", 0)
    #print("Subset rel performance", sum(subset_rel_perf) / len(subset_rel_perf))
