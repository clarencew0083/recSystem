# -*- coding: utf-8 -*-
"""
Created on 01 March 2019
@author: megan.woods

This script contains helper functions. Placed here to decultter other scripts.

It is called in
    - _04_algorithms
    - _05_main
"""
# Helper functions for the recommendation system__________________________________________
# ________________________________________________________________________________________
def unique(list1): 
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def find_ranks(performance_dict, return_sorted = False):
    """ Function to find rankings of the algorithms

        Parameters
        ---------
            performance_dict: dictionary
                performances calculated per algorithm
            return_sorted = boolean
                False: return ranks ordered by order of algorithms in calculate_accuracies function
                True: return ranks ordered from highest to lowest

        Returns
        -------
            dictionary, where keys are algorithms and values are ranks
    """
    perf = performance_dict.copy()
    ranks_dict = {key: rank for rank, key in enumerate(sorted(set(perf.values()), reverse=True), 1)}
    ranks = {k: ranks_dict[v] for k,v, in perf.items()} # unordered ranks

    if return_sorted == True:
        num = 1
        ranks_ordered = {}
        ranks_temp = ranks.copy()
        while num != len(ranks)+1:
            h_rank = min(ranks_temp.items(), key=lambda x: x[1]) # find key, value with highest rank
            ranks_ordered[h_rank[0]] = h_rank[1] # add key, value to new dictionary
            ranks_temp.pop(h_rank[0]) # remove key, value from temp dictionary
            num = num + 1 # update indicator
        return ranks_ordered
    else:
        return ranks

# OTHER___________________________________________________________________________________
# ________________________________________________________________________________________
def extract(myDict, keys = [], values = []):
    """ Function to get a subset of dictionary from a dictionary

        Parameters
        ----------
            myDict: dict
                the dictionary from which to extract
            keys: list
                names of keys to subset on
            values: list
                values to search for

        Returns
        -------
            subset of dictionary
    """
    if len(values) != 0:
        return dict((k, myDict[k]) for k, v in myDict.items() if v in values)
    if len(keys) != 0:
        return dict((k, myDict[k]) for k in keys if k in myDict)

def is_number(s):
    """ checks to see if data in file is a number or not
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def maybe_float(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s

# find all values in df that are in datasets
def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


