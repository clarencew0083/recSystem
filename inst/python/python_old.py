import create_models_bin_response as cr_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def recommend(df6, df7, df8, df9, df11, response11):

  threshold = 1


  response6 = 'target'

  data6_mse, data6_acc, data6_f1, data6_conf, data6_nrmse, data6_time, data6_precision, data6_recall, data6_nrmse2,\
    = cr_model.create_models_bin_response(df6, response6, 'heart')
  meta_features6 = cr_model.get_meta_features(df6, 'heart')


  response7 = 'yesno_bin'

  data7_mse, data7_acc, data7_f1, data7_conf, data7_nrmse, data7_time, data7_precision, data7_recall, data7_nrmse2,  \
         = cr_model.create_models_bin_response(df7, response7, 'spam7')
  meta_features7 = cr_model.get_meta_features(df7, 'spam7')


  response8 = 'Personal Loan'

  data8_mse, data8_acc, data8_f1, data8_conf, data8_nrmse, data8_time, data8_precision, data8_recall, data8_nrmse2, = \
    cr_model.create_models_bin_response(df8, response8, 'Bank_Personal_Loan')
  meta_features8 = cr_model.get_meta_features(df8, 'Bank_Personal_Loan')


  response9 = 'TenYearCHD'

  data9_mse, data9_acc, data9_f1, data9_conf, data9_nrmse, data9_time, data9_precision, data9_recall, data9_nrmse2,\
        = cr_model.create_models_bin_response(df9, response9, 'framingham')
  meta_features9 = cr_model.get_meta_features(df9, 'framingham')


  #response10 = 'CourseSuccess'

  #data10_mse, data10_acc, data10_f1, data10_conf, data10_nrmse, data10_time, data10_precision, data10_recall, data10_nrmse2,\
  #      = cr_model.create_models_bin_response(df10, response10, 'math_placement')
 # meta_features10 = cr_model.get_meta_features(df10, 'math_placement')



  data11_mse, data11_acc, data11_f1, data11_conf, data11_nrmse, data11_time, data11_precision, data11_recall, data11_nrmse2,\
    = cr_model.create_models_bin_response(df11, response11, 'data')
  meta_features11 = cr_model.get_meta_features(df11, 'data')

  # Load midterm data
  #df11 = pd.read_csv("C:\\Users\\c3_wi\\Desktop\\Python\\Data\\" + '699.csv')
  #df11.drop("TransactionID", axis=1, inplace=True)
  #response11 = 'isFraud'

  #data11_mse, data11_acc, data11_f1, data11_conf, data11_nrmse, data11_time, data11_precision, data11_recall, data11_nrmse2,\
  #       = cr_model.create_models_bin_response(df11, response11, 'credit_card_fraud')
  #meta_features11 = cr_model.get_meta_features(df11, 'credit_card_fraud')


  startscript = time.time()

  #nrmse is proba, nrmse2 is class
  frames_data = [data6_mse, data7_mse, data8_mse, data9_mse]
  combined_data = pd.concat(frames_data)

  #frames_nrmse_data = [data6_nrmse, data7_nrmse, data8_nrmse, data9_nrmse, data10_nrmse, data11_nrmse]
  #ombined_nrmse_data = pd.concat(frames_nrmse_data)

  #frames_nrmse2_data = [data6_nrmse2, data7_nrmse2, data8_nrmse2, data9_nrmse2, data10_nrmse2, data11_nrmse2]
  #combined_nrmse2_data = pd.concat(frames_nrmse2_data)

  #frames_f1_data = [data6_f1, data7_f1, data8_f1, data9_f1, data10_f1, data11_f1]
  #combined_f1_data = pd.concat(frames_f1_data)


  #frames_precision_data = [data6_precision, data7_precision, data8_precision, data9_precision, data10_precision, data11_precision]
  #combined_precision_data = pd.concat(frames_precision_data)

  frames_recall_data = [data6_recall, data7_recall, data8_recall, data9_recall, data11_recall]
  combined_recall_data = pd.concat(frames_recall_data)

  #frames_auc_data = [data6_roc_auc, data7_roc_auc, data8_roc_auc, data9_roc_auc, data10_roc_auc]
  #combined_auc_data = pd.concat(frames_auc_data)

  frames_time_data = [data6_time, data7_time, data8_time, data9_time, data11_time]
  combined_time_data = pd.concat(frames_time_data)




  frames_meta = [meta_features6, meta_features7, meta_features8, meta_features9,  meta_features11]
  combined_meta = pd.concat(frames_meta)



  # standard scaler centers and normalizes data
  sc = StandardScaler()
  X_train_std = sc.fit_transform(combined_meta)

  # Perform PCA
  m, n = X_train_std.shape

  S = (1 / m) * X_train_std.T @ X_train_std

  u, e, v = np.linalg.svd(S)

  tot = sum(e)

  var_exp = [(i / tot) for i in sorted(e, reverse=True)]

  cum_var_exp = np.cumsum(var_exp)

  plt.bar(range(1, n + 1), var_exp, alpha=0.5, align='center', label='individual explained variance', width=0.25)
  #plt.step(range(1, n + 1), cum_var_exp, where='mid', label='cumulative explained variance', c='green')
  plt.xlabel("Principal Component Index")
  plt.ylabel("Explained Variance Ratio")
  plt.legend(loc='center')
  plt.title('Meta Features: Variance Explained of Principal Components')
  plt.ylim(0, 1)
  #plt.show()

  combined_meta_copy = combined_meta.copy()

  combined_meta = sc.fit_transform(combined_meta)

  combined_meta = pd.DataFrame({
      'Rows': combined_meta[:, 0],
      'Columns': combined_meta[:, 1],
      'Rows-Cols Ratio': combined_meta[:, 2],
      'Number Discrete': combined_meta[:, 3],
      'Max num factors': combined_meta[:, 4],
      'Min num factors': combined_meta[:, 5],
      'Avg num factors': combined_meta[:, 6],
      'Number Continuous': combined_meta[:, 7],
      'Gradient-Avg': combined_meta[:, 8],
      'Gradient-Min': combined_meta[:, 9],
      'Gradient-Max': combined_meta[:, 10],
      'Gradient-Std': combined_meta[:, 11]},
      index=['Heart', 'Spam', 'Bank Personal Loan', 'Framingham', 'data'])

  # Perform dimension reduction on train data



  # 2 or 3 components for bin, 3 or 4 for con
  pca = PCA(n_components=3)
  #SVM, KNN, NB, SVR, RR, LR

  #metric = combined_nrmse2_data.copy()
  metric = combined_recall_data

  svm_pred = cr_model.meta_model(combined_meta, metric, 'SVM')
  knn_pred = cr_model.meta_model(combined_meta, metric, 'KNN')
  nb_pred = cr_model.meta_model(combined_meta, metric, 'NB')
  svr_pred = cr_model.meta_model(combined_meta, metric, 'SVR')
  rr_pred = cr_model.meta_model(combined_meta, metric, 'RR')
  lr_pred = cr_model.meta_model(combined_meta, metric, 'LR')

  frames_pred = [svm_pred, knn_pred, nb_pred, svr_pred, rr_pred, lr_pred]
  combined_pred = pd.concat(frames_pred, axis=1)


  #print("Actual Ranking \n", metric.loc['data'].rank(ascending=False), flush=True)
  print("Predicted Ranking \n", combined_pred.loc['data'].rank(ascending=False), flush=True)
  print("----------------------------------------------------------------------------------------")
  #print("Actual Max Recall:", metric.loc['data'].max(), metric.loc['data'].idxmax(), '\n', flush=True)
  print("Predicted Max Recall:", combined_pred.loc['data'].max(), combined_pred.loc['data'].idxmax(), '\n', flush=True)



  end = time.time()
  meta_time = (end - startscript)
  print("The time to build meta model is", meta_time, flush=True)
  


