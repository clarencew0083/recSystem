import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, mean_squared_error, average_precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


def create_models_bin_response(df, response, name, svr_threshold=0.5, ridge_threshold=0.5, lr_threshold=0.5, seed=1):
    y = df[response]
    y_min = min(y)
    y_max = max(y)

    df.drop(columns=response, axis=1, inplace=True)

    # standard scaler centers and normalizes data
    sc = StandardScaler()

    df = sc.fit_transform(df)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(df,
                                                                        y, test_size=0.4, random_state=seed,
                                                                        stratify=y)
    start_svm = time.time()
    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf', probability=True, gamma='scale', class_weight='balanced')

    # clf is for SVM
    clf.fit(X_train, y_train)

    svm_test_pred = clf.predict(X_test)
    svm_y_scores = clf.predict_proba(X_test)
    svm_mse = mean_squared_error(y_test, svm_y_scores[:, 1])
    svm_rmse = np.sqrt(svm_mse)
    svm_nrmse = svm_rmse / (y_max - y_min)
    svm_accuracy = metrics.accuracy_score(y_test, svm_test_pred)
    svm_f1 = f1_score(y_test, svm_test_pred)
    svm_precision = precision_score(y_test, svm_test_pred)
    svm_recall = recall_score(y_test, svm_test_pred)
    svm_conf = confusion_matrix(y_test, svm_test_pred)

    svm_mse2 = mean_squared_error(y_test, svm_test_pred)
    svm_rmse2 = np.sqrt(svm_mse2)
    svm_nrmse2 = svm_rmse2 / (y_max - y_min)


    # svm_avg_precis = average_precision_score(y_test, svm_y_scores[:, 1])
    # svm_fpr, svm_tpr, svm_threshold = roc_curve(y_test, svm_y_scores[:, 1])
    # svm_roc_auc = auc(svm_fpr, svm_tpr)

    end_svm = time.time()
    svm_time = (end_svm - start_svm)

    # KNN Classifier
    start_knn = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)

    # Fit KNN Classifier
    knn.fit(X_train, y_train)

    knn_y_scores = knn.predict_proba(X_test)

    knn_test_pred = knn.predict(X_test)

    knn_mse = mean_squared_error(y_test, knn_y_scores[:, 1])
    knn_rmse = np.sqrt(knn_mse)
    knn_nrmse = knn_rmse / (y_max - y_min)
    knn_accuracy = metrics.accuracy_score(y_test, knn_test_pred)

    knn_mse2 = mean_squared_error(y_test, knn_test_pred)
    knn_rmse2 = np.sqrt(knn_mse2)
    knn_nrmse2 = knn_rmse2 / (y_max - y_min)

    knn_f1 = f1_score(y_test, knn_test_pred)
    knn_precision = precision_score(y_test, knn_test_pred)
    knn_recall = recall_score(y_test, knn_test_pred)
    knn_conf = confusion_matrix(y_test, knn_test_pred)

    # knn_fpr, knn_tpr, knn_threshold = roc_curve(y_test, knn_y_scores[:, 1])

    # knn_roc_auc = auc(knn_fpr, knn_tpr)

    end_knn = time.time()
    knn_time = (end_knn - start_knn)

    # Naive Bayes

    start_nb = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_y_scores = nb.predict_proba(X_test)

    # nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, nb_y_scores[:, 1])
    # nb_roc_auc = auc(nb_fpr, nb_tpr)

    nb_predictions = nb.predict(X_test)
    nb_mse = mean_squared_error(y_test, nb_y_scores[:, 1])
    nb_rmse = np.sqrt(nb_mse)
    nb_nrmse = nb_rmse / (y_max - y_min)
    nb_accuracy = metrics.accuracy_score(y_test, nb_predictions)

    nb_mse2 = mean_squared_error(y_test, nb_predictions)
    nb_rmse2 = np.sqrt(nb_mse2)
    nb_nrmse2 = nb_rmse2 / (y_max - y_min)

    nb_f1 = f1_score(y_test, nb_predictions)
    nb_conf = confusion_matrix(y_test, nb_predictions)
    nb_precision = precision_score(y_test, nb_predictions)
    nb_recall = recall_score(y_test, nb_predictions)

    end_nb = time.time()
    nb_time = (end_nb - start_nb)

    ##MODEL SUPPORT VECTOR REGRESSION
    start_svr = time.time()
    # Support Vector Regression
    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')  # default settings
    svr.fit(X_train, y_train)
    y_pred_final_SVR = svr.predict(X_test)
    y_pred_final_SVR[y_pred_final_SVR < 0] = 0
    y_pred_final_SVR[y_pred_final_SVR > 1] = 1
    svr_proba = y_pred_final_SVR.copy()

    # y_pred_final_SVR_round = y_pred_final_SVR.round()

    # Threshold for class boundary
    y_pred_final_SVR[y_pred_final_SVR < svr_threshold] = 0
    y_pred_final_SVR[y_pred_final_SVR >= svr_threshold] = 1
    y_pred_final_SVR_round = y_pred_final_SVR

    # svr_fpr, svr_tpr, svr_threshold = roc_curve(y_test, svr_proba)

    # svr_roc_auc = auc(svr_fpr, svr_tpr)

    svr_mse = mean_squared_error(y_test, svr_proba)
    svr_rmse = np.sqrt(svr_mse)
    svr_nrmse = svr_rmse / (y_max - y_min)

    svr_mse2 = mean_squared_error(y_test, y_pred_final_SVR)
    svr_rmse2 = np.sqrt(svr_mse2)
    svr_nrmse2 = svr_rmse2 / (y_max - y_min)

    svr_accuracy = metrics.accuracy_score(y_test, y_pred_final_SVR_round)
    svr_f1 = f1_score(y_test, y_pred_final_SVR_round)
    svr_precision = precision_score(y_test, y_pred_final_SVR_round)
    svr_recall = recall_score(y_test, y_pred_final_SVR_round)
    svr_conf = confusion_matrix(y_test, y_pred_final_SVR_round)

    end_svr = time.time()
    svr_time = (end_svr - start_svr)

    # Ridge Regression
    start_ridge = time.time()
    ridge = linear_model.Ridge()

    ridge.fit(X_train, y_train)
    y_pred_final_ridge = ridge.predict(X_test)
    y_pred_final_ridge[y_pred_final_ridge < 0] = 0
    y_pred_final_ridge[y_pred_final_ridge > 1] = 1

    ridge_proba = y_pred_final_ridge.copy()
    # y_pred_final_ridge_round = y_pred_final_ridge.round()

    # Threshold
    y_pred_final_ridge[y_pred_final_ridge < ridge_threshold] = 0
    y_pred_final_ridge[y_pred_final_ridge >= ridge_threshold] = 1
    y_pred_final_ridge_round = y_pred_final_ridge

    # ridge_fpr, ridge_tpr, ridge_threshold = roc_curve(y_test, ridge_proba)

    # ridge_roc_auc = auc(ridge_fpr, ridge_tpr)

    ridge_mse = mean_squared_error(y_test, ridge_proba)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_nrmse = ridge_rmse / (y_max - y_min)
    ridge_accuracy = metrics.accuracy_score(y_test, y_pred_final_ridge_round)

    ridge_mse2 = mean_squared_error(y_test, y_pred_final_ridge)
    ridge_rmse2 = np.sqrt(ridge_mse2)
    ridge_nrmse2 = ridge_rmse2 / (y_max - y_min)

    ridge_f1 = f1_score(y_test, y_pred_final_ridge_round)
    ridge_precision = precision_score(y_test, y_pred_final_ridge_round)
    ridge_recall = recall_score(y_test, y_pred_final_ridge_round)
    ridge_conf = confusion_matrix(y_test, y_pred_final_ridge_round)

    end_ridge = time.time()
    ridge_time = (end_ridge - start_ridge)

    # LR
    start_lr = time.time()
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_final_lr = lr.predict(X_test)

    y_pred_final_lr[y_pred_final_lr < 0] = 0
    y_pred_final_lr[y_pred_final_lr > 1] = 1
    # y_pred_final_lr_round = y_pred_final_lr.round()
    lr_proba = y_pred_final_lr.copy()

    # Threshold
    y_pred_final_lr[y_pred_final_lr < lr_threshold] = 0
    y_pred_final_lr[y_pred_final_lr >= lr_threshold] = 1
    y_pred_final_lr_round = y_pred_final_lr

    # lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, lr_proba)

    # lr_roc_auc = auc(lr_fpr, lr_tpr)

    lr_mse = mean_squared_error(y_test, lr_proba)
    lr_rmse = np.sqrt(lr_mse)
    lr_nrmse = lr_rmse / (y_max - y_min)
    lr_accuracy = metrics.accuracy_score(y_test, y_pred_final_lr_round)

    lr_mse2 = mean_squared_error(y_test, y_pred_final_lr)
    lr_rmse2 = np.sqrt(lr_mse2)
    lr_nrmse2 = lr_rmse2 / (y_max - y_min)

    lr_f1 = f1_score(y_test, y_pred_final_lr_round)
    lr_precision = precision_score(y_test, y_pred_final_lr_round)
    lr_recall = recall_score(y_test, y_pred_final_lr_round)
    lr_conf = confusion_matrix(y_test, y_pred_final_lr_round)

    end_lr = time.time()
    lr_time = (end_lr - start_lr)

    data = {
        'Model': [name],
        'SVM': [svm_rmse],
        'KNN': [knn_rmse],
        'NB': [nb_rmse],
        'SVR': [svr_rmse],
        "RR": [ridge_rmse],
        'LR': [lr_rmse],
    }

    data = pd.DataFrame(data)
    data.set_index('Model', inplace=True)

    data2 = {
        'Model': [name],
        'SVM': [svm_accuracy],
        'KNN': [knn_accuracy],
        'NB': [nb_accuracy],
        'SVR': [svr_accuracy],
        "RR": [ridge_accuracy],
        'LR': [lr_accuracy],
    }

    data2 = pd.DataFrame(data2)
    data2.set_index('Model', inplace=True)

    data3 = {
        'Model': [name],
        'SVM': [svm_f1],
        'KNN': [knn_f1],
        'NB': [nb_f1],
        'SVR': [svr_f1],
        "RR": [ridge_f1],
        'LR': [lr_f1],
    }

    data3 = pd.DataFrame(data3)
    data3.set_index('Model', inplace=True)

    data4 = {
        'Model': [name],
        'SVM': [svm_conf],
        'KNN': [knn_conf],
        'NB': [nb_conf],
        'SVR': [svr_conf],
        "RR": [ridge_conf],
        'LR': [lr_conf],
    }

    data4 = pd.DataFrame(data4)
    data4.set_index('Model', inplace=True)

    data5 = {
        'Model': [name],
        'SVM': [svm_nrmse],
        'KNN': [knn_nrmse],
        'NB': [nb_nrmse],
        'SVR': [svr_nrmse],
        "RR": [ridge_nrmse],
        'LR': [lr_nrmse],
    }

    data5 = pd.DataFrame(data5)
    data5.set_index('Model', inplace=True)

    data6 = {
        'Model': [name],
        'SVM': [svm_time],
        'KNN': [knn_time],
        'NB': [nb_time],
        'SVR': [svr_time],
        "RR": [ridge_time],
        'LR': [lr_time],
    }

    data6 = pd.DataFrame(data6)
    data6.set_index('Model', inplace=True)

    data7 = {
        'Model': [name],
        'SVM': [svm_precision],
        'KNN': [knn_precision],
        'NB': [nb_precision],
        'SVR': [svr_precision],
        "RR": [ridge_precision],
        'LR': [lr_precision],
    }

    data7 = pd.DataFrame(data7)
    data7.set_index('Model', inplace=True)

    data8 = {
        'Model': [name],
        'SVM': [svm_recall],
        'KNN': [knn_recall],
        'NB': [nb_recall],
        'SVR': [svr_recall],
        "RR": [ridge_recall],
        'LR': [lr_recall],
    }

    data8 = pd.DataFrame(data8)
    data8.set_index('Model', inplace=True)

    data9 = {
        'Model': [name],
        'SVM': [svm_nrmse2],
        'KNN': [knn_nrmse2],
        'NB': [nb_nrmse2],
        'SVR': [svr_nrmse2],
        "RR": [ridge_nrmse2],
        'LR': [lr_nrmse2],
    }

    data9 = pd.DataFrame(data9)
    data9.set_index('Model', inplace=True)

    return data, data2, data3, data4, data5, data6, data7, data8, data9


def create_models_con_response(df, response, name):
    y = df[response]
    y_min = min(y)
    y_max = max(y)
    df.drop(columns=response, axis=1, inplace=True)

    # standard scaler centers and normalizes data
    sc = StandardScaler()

    df = sc.fit_transform(df)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(df,
                                                                        y, test_size=0.4, random_state=1)

    y_copy = y_test.copy()
    y_mean = y.mean()
    y_copy[y_copy < y_mean] = 0
    y_copy[y_copy >= y_mean] = 1

    ##MODEL SUPPORT VECTOR REGRESSION

    # Support Vector Regression
    start_svr = time.time()
    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')  # default settings
    svr.fit(X_train, y_train)
    y_pred_final_SVR = svr.predict(X_test)
    svr_mse = mean_squared_error(y_test, y_pred_final_SVR)
    svr_rmse = np.sqrt(svr_mse)
    svr_nrmse = svr_rmse / (y_max - y_min)

    y_pred_final_SVR_copy = y_pred_final_SVR.copy()
    y_pred_final_SVR_copy[y_pred_final_SVR_copy < y_mean] = 0
    y_pred_final_SVR_copy[y_pred_final_SVR_copy >= y_mean] = 1

    svr_f1 = f1_score(y_copy, y_pred_final_SVR_copy)

    end_svr = time.time()
    svr_time = (end_svr - start_svr)

    # RR
    start_ridge = time.time()
    ridge = linear_model.Ridge()

    ridge.fit(X_train, y_train)
    y_pred_final_ridge = ridge.predict(X_test)

    ridge_mse = mean_squared_error(y_test, y_pred_final_ridge)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_nrmse = ridge_rmse / (y_max - y_min)

    y_pred_final_ridge_copy = y_pred_final_ridge.copy()
    y_pred_final_ridge_copy[y_pred_final_ridge_copy < y_mean] = 0
    y_pred_final_ridge_copy[y_pred_final_ridge_copy >= y_mean] = 1

    ridge_f1 = f1_score(y_copy, y_pred_final_ridge_copy)

    end_ridge = time.time()
    ridge_time = (end_ridge - start_ridge)

    # LR
    start_lr = time.time()
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_final_lr = lr.predict(X_test)

    lr_mse = mean_squared_error(y_test, y_pred_final_lr)
    lr_rmse = np.sqrt(lr_mse)
    lr_nrmse = lr_rmse / (y_max - y_min)

    y_pred_final_lr_copy = y_pred_final_lr.copy()
    y_pred_final_lr_copy[y_pred_final_lr_copy < y_mean] = 0
    y_pred_final_lr_copy[y_pred_final_lr_copy >= y_mean] = 1

    lr_f1 = f1_score(y_copy, y_pred_final_lr_copy)

    end_lr = time.time()
    lr_time = (end_lr - start_lr)

    data = {
        'Model': [name],
        'SVR': [svr_rmse],
        "RR": [ridge_rmse],
        'LR': [lr_rmse],

    }

    data = pd.DataFrame(data)
    data.set_index('Model', inplace=True)

    data2 = {
        'Model': [name],
        'SVR': [svr_f1],
        "RR": [ridge_f1],
        'LR': [lr_f1],

    }

    data2 = pd.DataFrame(data2)
    data2.set_index('Model', inplace=True)

    data3 = {
        'Model': [name],
        'SVR': [svr_nrmse],
        "RR": [ridge_nrmse],
        'LR': [lr_nrmse],

    }

    data3 = pd.DataFrame(data3)
    data3.set_index('Model', inplace=True)

    data4 = {
        'Model': [name],
        'SVR': [svr_time],
        "RR": [ridge_time],
        'LR': [lr_time],
    }

    data4 = pd.DataFrame(data4)
    data4.set_index('Model', inplace=True)

    return data, data2, data3, data4


def separate_cont_and_discrete(df):
    """Question: How do we know if the data is discrete?
            12 int64 and 1 float64 column

            number of unique values: 228, 9, 97, 120, 108, 98, 82, 56, 195, 204, 143, 172, 10
            for  i in temp.columns: print(238/temp[i].nunique())
                1  26  2.5  2  2.2  2.4  2.9  4.3  1  1  1.6  1.4  23

        """
    sub_df = df.select_dtypes(include=["number"])
    for i in sub_df.columns:
        if i == "idp":
            print(i)

    # sub_df = sub_df.drop(self.target, 1).copy()
    num_rows = len(sub_df)
    indicator = int(num_rows < 30)
    threshold = 0.25 * (1 - indicator) + 0.5 * indicator  # threshold = 0.25 if numRows >= 30, 0.5 if numRows < 30
    # print("Number of rows: ", num_rows, "  Indicator: ", indicator, "  Threshold: ", threshold)

    #            test_name = "Baseball"
    # current method to determine if discrete:
    discrete = []
    for i in sub_df.columns:
        value = "Continuous"
        #                if self.name == test_name:
        #                    print(i)
        #                    print(num_rows)
        #                    print("Num unique", sub_df[i].nunique())
        if sub_df[i].nunique() / num_rows <= threshold:
            value = "Discrete"
            discrete.append(i)
    #                if self.name == test_name:
    #                    print(i, ": ", sub_df[i].nunique()/num_rows, " ", value)
    #                    print("\n")

    discrete_df = sub_df[[x for x in sub_df.columns if x in discrete]]
    continuous_df = sub_df[[x for x in sub_df.columns if x not in discrete]]

    #            print("Continuous: ", continuous_df.columns)
    #            print("Discrete: ", discrete)
    #            print("\n")
    #            print(self.name)
    #            print("Number of rows: ", num_rows)
    #            print("Threshold: ", threshold)
    #            print("\n", self.original_data[sub_df.columns].head(10), "\n")
    #            print("\n", self.original_data[sub_df.columns].tail(10), "\n")
    return continuous_df, discrete_df


# Fit a rbf kernel SVM
# svc = SVR(kernel='rbf', epsilon=0.3, gamma=0.7, C=64)
# svc.fit(X_train, y_train)

# Get prediction for a point X_test using train SVM, svc
def get_pred(model, data, train_data):
    def RBF(x, z, gamma, axis=None):
        return np.exp((-gamma * np.linalg.norm(x - z, axis=axis) ** 2))

    A = []
    # Loop over all support vectors to calculate K(Xi, X_test), for Xi belongs to the set of support vectors
    # Replace 1 with actual gamma
    for x in model.support_vectors_:
        A.append(RBF(x, data, 1 / (5 * np.std(train_data))))

    A = np.array(A)
    return np.sum(model._dual_coef_ * A) + model.intercept_


def get_meta_features(df, name):
    continuous_df, discrete_df = separate_cont_and_discrete(df)
    num_continuous = len(continuous_df.columns)
    num_discrete = len(discrete_df.columns)

    num_df = numeric_df = df.select_dtypes(include=["number"])
    gradient = np.gradient(num_df.values)
    horizontal_gradient = gradient[1]

    # find greatest number of unique values in discrete column
    disc_num_unique = []
    if num_discrete != 0:
        for col in discrete_df.columns:
            disc_num_unique.append(len(discrete_df[col].unique()))

            if len(disc_num_unique) != 0:
                max_disc_num_unique = max(disc_num_unique)
                min_disc_num_unique = min(disc_num_unique)
                avg_disc_num_unique = sum(disc_num_unique) / float(len(disc_num_unique))
            else:
                max_disc_num_unique = 0
                min_disc_num_unique = 0
                avg_disc_num_unique = 0
    else:
        max_disc_num_unique = 0
        min_disc_num_unique = 0
        avg_disc_num_unique = 0

    num_columns_in_modified_original_df = len(df.columns)

    meta_features = {
        "Data": name,
        "Rows": len(df.index),
        "Columns": num_columns_in_modified_original_df,
        "Rows-Cols Ratio": len(df.index) / num_columns_in_modified_original_df,
        "Number Discrete": num_discrete,
        "Max num factors": max_disc_num_unique,
        "Min num factors": min_disc_num_unique,
        "Avg num factors": avg_disc_num_unique,
        "Number Continuous": num_continuous,
        "Gradient-Avg": horizontal_gradient.mean(),
        "Gradient-Min": horizontal_gradient.min(),
        "Gradient-Max": horizontal_gradient.max(),
        "Gradient-Std": horizontal_gradient.std()
    }

    meta_features = pd.DataFrame(meta_features, index=[0])
    meta_features.set_index('Data', inplace=True)
    return meta_features


def meta_model(combined_meta, metric_df, algorithm_name):
    loo = LeaveOneOut()
    loo.get_n_splits(combined_meta)

    idx = metric_df.columns.get_loc(algorithm_name)
    m, n = combined_meta.shape
    pca = PCA(n_components=3)
    y_pred = np.zeros(shape=(m, 1))

    for train_index, test_index in loo.split(combined_meta):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = combined_meta.iloc[train_index, :], combined_meta.iloc[test_index, :]
        pca_train_data = pca.fit_transform(X_train)
        pca_test_data = pca.transform(X_test)
        y_train, y_test = metric_df.iloc[train_index, idx], metric_df.iloc[test_index, idx]

        # Calculate actual gamma values to test
        model = SVR(C=1, epsilon=0.1, gamma='scale')

        # model = linear_model.LinearRegression()
        model.fit(pca_train_data, y_train)

        y_pred[test_index] = model.predict(pca_test_data)

        # Uncomment the next two lines to manually get prediction.
        #print(get_pred(model, pca_test_data, pca_train_data))
        #print(y_pred[test_index])  # The same oputput by both

        # print("Train Data", X_train, X_test, "\n Response \n", y_train, y_test)
        # print(y_test, '\n')

        # y_pred = pd.DataFrame({
        #   algorithm_name: y_pred, }, index=y_test.index)

    data = pd.DataFrame(y_pred, index=metric_df.index)
    data.columns = [algorithm_name]

    return data


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, name):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title(name + " Precision Recall Scores vs Decision Threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.savefig(name + '.png')
    plt.show()


def plot_f1_vs_threshold(f1, thresholds, name):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title(name + " F1 Score vs Decision Threshold")
    plt.plot(thresholds, f1[:-1], "b--", label="F1 Score")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.savefig(name + '_f1.png')
    plt.show()


def create_models_response(df, response, name):
    y = df[response]
    y_min = min(y)
    y_max = max(y)

    df.drop(columns=response, axis=1, inplace=True)

    # standard scaler centers and normalizes data
    sc = StandardScaler()

    df = sc.fit_transform(df)
    custom_scorer = make_scorer(norm_rmse, greater_is_better=False)
    scoring = {'precision': 'precision',
               'recall': 'recall',
               'f1_score': 'f1',
               'nrmse': custom_scorer}
    start_svm = time.time()
    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf', probability=False, gamma='scale', class_weight='balanced')

    scores_svm = cross_validate(clf, df, y, scoring=scoring, cv=5, return_train_score=True)

    knn = KNeighborsClassifier(n_neighbors=5)
    # Fit KNN Classifier
    scores_knn = cross_validate(knn, df, y, scoring=scoring, cv=5, return_train_score=True)

    # Naive Bayes

    nb = GaussianNB()
    scores_nb = cross_validate(knn, df, y, scoring=scoring, cv=5, return_train_score=True)

    custom_scorer = make_scorer(norm_rmse, greater_is_better=False)
    con_scoring = {'nrmse': custom_scorer}

    ##MODEL SUPPORT VECTOR REGRESSION

    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')  # default settings
    scores_svr = cross_validate(svr, df, y, scoring=con_scoring, cv=5, return_train_score=True)
    svr_list_f1, svr_list_precision, svr_list_recall = custom_cross_validate(df, y, 'svr', svr)

    svr_f1_score = np.asarray(svr_list_f1)
    svr_precision = np.asarray(svr_list_precision)
    svr_recall = np.asarray(svr_list_recall)

    # Ridge Regression

    ridge = linear_model.Ridge()
    scores_ridge = cross_validate(ridge, df, y, scoring=con_scoring, cv=5, return_train_score=True)
    ridge_list_f1, ridge_list_precision, ridge_list_recall = custom_cross_validate(df, y, 'ridge', ridge)

    ridge_f1_score = np.asarray(ridge_list_f1)
    ridge_precision = np.asarray(ridge_list_precision)
    ridge_recall = np.asarray(ridge_list_recall)

    # LR
    lr = linear_model.LinearRegression()
    scores_lr = cross_validate(lr, df, y, scoring=con_scoring, cv=5, return_train_score=True)
    lr_list_f1, lr_list_precision, lr_list_recall = custom_cross_validate(df, y, 'lr', lr)

    lr_f1_score = np.asarray(lr_list_f1)
    lr_precision = np.asarray(lr_list_precision)
    lr_recall = np.asarray(lr_list_recall)

    data = {
        'Model': [name],
        'SVM': scores_svm['test_nrmse'].mean(),
        'KNN': scores_knn['test_nrmse'].mean(),
        'NB': scores_nb['test_nrmse'].mean(),
        'SVR': scores_svr['test_nrmse'].mean(),
        "RR": scores_ridge['test_nrmse'].mean(),
        'LR': scores_lr['test_nrmse'].mean(),
    }

    data = pd.DataFrame(data)
    data.set_index('Model', inplace=True)

    data_std = {
        'Model': [name],
        'SVM': scores_svm['test_nrmse'].std(),
        'KNN': scores_knn['test_nrmse'].std(),
        'NB': scores_nb['test_nrmse'].std(),
        'SVR': scores_svr['test_nrmse'].std(),
        "RR": scores_ridge['test_nrmse'].std(),
        'LR': scores_lr['test_nrmse'].std(),
    }

    data_std = pd.DataFrame(data_std)
    data_std.set_index('Model', inplace=True)

    data2 = {
        'Model': [name],
        'SVM': scores_svm['test_f1_score'].mean(),
        'KNN': scores_knn['test_f1_score'].mean(),
        'NB': scores_nb['test_f1_score'].mean(),
        'SVR': svr_f1_score.mean(),
        "RR": ridge_f1_score.mean(),
        'LR': lr_f1_score.mean(),
    }

    data2 = pd.DataFrame(data2)
    data2.set_index('Model', inplace=True)

    data2_std = {
        'Model': [name],
        'SVM': scores_svm['test_f1_score'].std(),
        'KNN': scores_knn['test_f1_score'].std,
        'NB': scores_nb['test_f1_score'].std(),
        'SVR': svr_f1_score.std(),
        "RR": ridge_f1_score.std(),
        'LR': lr_f1_score.std(),
    }

    data2_std = pd.DataFrame(data2_std)
    data2_std.set_index('Model', inplace=True)

    data3 = {
        'Model': [name],
        'SVM': scores_svm['test_precision'].mean(),
        'KNN': scores_knn['test_precision'].mean(),
        'NB': scores_nb['test_precision'].mean(),
        'SVR': svr_precision.mean(),
        "RR": ridge_precision.mean(),
        'LR': lr_precision.mean(),
    }

    data3 = pd.DataFrame(data3)
    data3.set_index('Model', inplace=True)

    data3_std = {
        'Model': [name],
        'SVM': scores_svm['test_precision'].std(),
        'KNN': scores_knn['test_precision'].std(),
        'NB': scores_nb['test_precision'].std(),
        'SVR': svr_precision.std(),
        "RR": ridge_precision.std(),
        'LR': lr_precision.std(),
    }

    data3_std = pd.DataFrame(data3_std)
    data3_std.set_index('Model', inplace=True)

    data4 = {
        'Model': [name],
        'SVM': scores_svm['test_recall'].mean(),
        'KNN': scores_knn['test_recall'].mean(),
        'NB': scores_nb['test_recall'].mean(),
        'SVR': svr_recall.mean(),
        "RR": ridge_recall.mean(),
        'LR': lr_recall.mean(),
    }

    data4 = pd.DataFrame(data4)
    data4.set_index('Model', inplace=True)

    data4_std = {
        'Model': [name],
        'SVM': scores_svm['test_recall'].std(),
        'KNN': scores_knn['test_recall'].std(),
        'NB': scores_nb['test_recall'].std(),
        'SVR': svr_recall.std(),
        "RR": ridge_recall.std(),
        'LR': lr_recall.std(),
    }

    data4_std = pd.DataFrame(data4_std)
    data4_std.set_index('Model', inplace=True)

    return data, data_std, data2, data2_std, data3, data3_std,  data4, data4_std


def norm_rmse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return -rmse


def custom_recall(y_test, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    calc_recall = recall_score(y_test, y_pred)

    return calc_recall


def custom_precision(y_test, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    calc_precision = precision_score(y_test, y_pred)

    return calc_precision


def custom_f1(y_test, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    calc_f1 = f1_score(y_test, y_pred)

    return calc_f1


def custom_cross_validate(df, y, algorithm, model):
    list_f1 = []
    list_precision = []
    list_recall = []

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in skf.split(df, y):

        X_train, X_test, y_train, y_test = df[train_index], df[test_index], y[train_index], y[test_index]

        if algorithm == 'svr':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            list_f1.append(custom_f1(y_test, y_pred))

            list_precision.append(custom_precision(y_test, y_pred))

            list_recall.append(custom_recall(y_test, y_pred))

        if algorithm == 'lr':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            list_f1.append(custom_f1(y_test, y_pred))

            list_precision.append(custom_precision(y_test, y_pred))

            list_recall.append(custom_recall(y_test, y_pred))

        if algorithm == 'ridge':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            list_f1.append(custom_f1(y_test, y_pred))

            list_precision.append(custom_precision(y_test, y_pred))

            list_recall.append(custom_recall(y_test, y_pred))

    return list_f1, list_precision, list_recall



