# -*- coding: utf-8 -*-
"""
Created on 01 March 2019
@author: megan.woods

This script is used to set up a single dataset. It
    - Loads the dataset
    - Preprocesses
    - Determines target column
    - Creates training and testing sets
    - Finds meta-features

It is called in
    - _05_main

"""
import os
import numpy as np
import pandas as pd

#from pydataset import data as pydata
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import _01_constants as constants

# Prepare the data________________________________________________________________________
# ****************************************************************************************
# ****************************************************************************************
class Prep_Data():
    """ For setting up a new data instance

        Parameters
        ----------
            name: string
                a string to identify this dataset
            target: string
                name of target column to predict
            directory: string
                location of csv file
            csv_file: string
                name of data to load
            dataframe: dataframe
                already loaded dataframe to pass in (optional)
            train_test: binary
                whether or not to split into training and testings sets

    """
    def __init__(self, name, target_col, dataframe, train_test=True):
        # init functions__________________________________________________________________
        # ________________________________________________________________________________

        self.dataframe = dataframe

        def compute_test_train(target, df):
            """ Function that computes the train and test datasets
            """
            sc = StandardScaler()
            x_df = df.drop(target,1)
            y_df = df[target]
            x_df_scaled = sc.fit_transform(x_df)
            X_train, X_test, Y_train, Y_test = train_test_split(x_df_scaled, y_df, test_size=0.4, random_state=1, stratify=y_df)
            return X_train, X_test, Y_train, Y_test

        def read_documentation():
            os.chdir(constants.documentation_dir)
            filename = name+".txt"
            if os.path.isfile(filename):
                file = open(filename, "r", encoding="utf-8")
                contents = file.read()
                file.close()
            else:
                contents = "DNE"
            os.chdir(constants.current_dir)
            return contents

        def determine_int_vs_float(df):
            float_col_list = []
            int_col_list = []
            
            for col in df.columns:
                if all(isinstance(x,float) for x in df[col])==True:
                    float_col_list.append(col)
                elif all(isinstance(x,int) for x in df[col])==True:
                    int_col_list.append(col)
            float_df = df[float_col_list]
            int_df = df[int_col_list]
            return float_df, int_df

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

            sub_df = sub_df.drop(self.target,1).copy()
            num_rows = len(sub_df)
            indicator = int(num_rows<30)
            threshold = 0.25*(1-indicator) + 0.5*indicator #threshold = 0.25 if numRows >= 30, 0.5 if numRows < 30
            #print("Number of rows: ", num_rows, "  Indicator: ", indicator, "  Threshold: ", threshold)


#            test_name = "Baseball"
            # current method to determine if discrete:
            discrete = []
            for i in sub_df.columns:
                value = "Continuous"
#                if self.name == test_name:
#                    print(i)
#                    print(num_rows)
#                    print("Num unique", sub_df[i].nunique())
                if sub_df[i].nunique()/num_rows <= threshold:
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

        def set_target(df):
            """ Set the target, unless already specified
            """
            number_list = list(df.select_dtypes(include=["number"]).columns)
            unique_values = list(df[number_list].nunique())
            max_value = max(unique_values)
            idx = unique_values.index(max_value)
            target = number_list[idx]
            return target

#        def unique(list1): 
#            # intilize a null list 
#            unique_list = [] 
#            # traverse for all elements 
#            for x in list1: 
#                # check if exists in unique_list or not 
#                if x not in unique_list: 
#                    unique_list.append(x) 
#            return unique_list
        
        # init variables__________________________________________________________________
        # ________________________________________________________________________________
        self.name = name
        #self.documentation = read_documentation()
        self.remove = False

        # Load data
        
        temp_df = self.dataframe
        # Preprocessing___________________________________________________________________
        temp_df = temp_df.dropna(1,how="all").dropna(0,how="any")



        cols = temp_df.columns.copy()
        for i in cols:
            # if column is boolean
            if temp_df[i].dtype.name=="bool":
                # change values to 0 and 1
                temp_df[i] = temp_df[i].astype(int)
            # drop columns that have the exact same input for each row
            if temp_df[i].nunique() == 1:
                temp_df = temp_df.drop(i,1)
            # drop columns that serve as an index column
            elif list(temp_df.index) == list(temp_df[i]):
                temp_df = temp_df.drop(i,1)

        # drop object columns that have all unique values
        object_cols = temp_df.select_dtypes(include=["object"]).columns
        df = temp_df.drop((i for i in object_cols if len(temp_df[i].unique())==len(temp_df[i])),1)

        # we need at least 3 numeric columns (including the target column) in order to take a gradient
        temp_num_cols = temp_df.select_dtypes(include=["number"]).columns
        if len(temp_num_cols) <= 2:
            self.remove = True
        elif len(temp_df) < constants.min_rows:
            self.remove = True
#        elif len(temp_df.columns) < min_cols:
#            self.remove = True
        else:
            num_columns_in_modified_original_df = len(df.columns)
            # set target column
            if (target_col == "") or (target_col not in temp_num_cols):
                #self.target = set_target(df)
                self.target = target_col
            else:
                self.target = target_col
            target = self.target
            self.original_data = df.copy()

            # label encode response
            le = preprocessing.LabelEncoder()

            df[str(target)] = le.fit_transform(df[str(target)])
                      
#            continuous_df, discrete_df = determine_int_vs_float(df)
            continuous_df, discrete_df = separate_cont_and_discrete(df)
            num_continuous = len(continuous_df.columns)
            num_discrete = len(discrete_df.columns)

            # find greatest number of unique values in discrete column
            disc_num_unique = []
            for col in discrete_df.columns:
                disc_num_unique.append(len(discrete_df[col].unique()))

            if len(disc_num_unique) != 0:
                max_disc_num_unique = max(disc_num_unique)
                min_disc_num_unique = min(disc_num_unique)
                avg_disc_num_unique = sum(disc_num_unique)/float(len(disc_num_unique))
            else:
                max_disc_num_unique = 0
                min_disc_num_unique = 0
                avg_disc_num_unique = 0            
            
            # one hot encoding for categorical variables
            df = pd.get_dummies(df,drop_first=True) #dtype = "float64"

            if len(continuous_df.columns) != 0:
                df[continuous_df.columns] = (pd.DataFrame(#data=min_max_scaler.fit_transform(continuous_df),
                                                          data=preprocessing.scale(continuous_df),
                                                          index=continuous_df.index,
                                                          columns=continuous_df.columns))
            self.cleaned_data = df
            self.numeric_df = df.select_dtypes(include=["number"]).drop(target,1)
            num_df = self.numeric_df
            self.num_pred_cols = len(num_df.columns)
            
            gradient = np.gradient(num_df.values) # do we compute the gradient on the numeric and hot-encoded?
            horizontal_gradient = gradient[1] # differences computed per row - pretty sure we want this one
            
            #print(horizontal_gradient)
            #vertical_gradient = gradient[0] # differences computed per column

            meta_features = {
                             "Rows": len(df.index),
                             "Columns": num_columns_in_modified_original_df,
                             "Rows-Cols Ratio": len(df.index)/num_columns_in_modified_original_df,
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

            meta_features = pd.DataFrame(data=[[v for v in meta_features.values()]],
                                         columns=[k for k in meta_features.keys()],
                                         index=[name])
            self.meta_features = meta_features

            # Training
            if train_test:
                self.X_train, self.X_test, self.y_train, self.y_test = compute_test_train(df=df,target=target)
                self.num_train = len(self.X_train)
                self.num_test  = len(self.X_test)
