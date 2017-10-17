from rolling_forecast import rolling_grid_search_ML
import pandas as pd
import numpy as np
from pandas import *
from numpy import *
from sklearn import svm
from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
import collections

def rmse(actual_lst,pred_lst):
    sum_sq_error = 0
    len_lst = len(actual_lst)
    for actual,pred in zip(actual_lst,pred_lst):
        sum_sq_error += (actual - pred)**2
    return (sum_sq_error/len_lst)**(0.5)
def crit_min(score_lst):
    min_val = score_lst[0]
    min_index = 0
    counter = 0
    for score in score_lst:
        if (score < min_val):
            min_index = counter
            min_val = score
        counter += 1
    return (min_index,min_val)

if __name__ == '__main__':
    data = pd.read_csv("../../../data/all_data.csv")
    data.set_index("DATE", inplace = True)
    var_lst = data.columns.tolist()
    for var in var_lst:
        data[var] = data[var].astype(float)
    var_lst = data.columns.tolist()
    num_lags = 1
    for var in var_lst:
        data[var] = data[var].astype(float)
        for lag in range(1,num_lags + 1):
            col_name = "L"+str(lag)+"."+str(var)
            data[col_name] = data[var].shift(-lag)
    data.dropna(axis=0, how='any', inplace = True)
    
    # Generate a list of labels and features
    import itertools
    endog_lst = ["US_EU","US_UK","JP_US"]
    _predictor_lst = ["L1.US_EU","L1.US_UK","L1.JP_US","L1.Oil_Price",
                      "L1.US_EU_LIBOR","L1.US_UK_LIBOR","L1.US_JP_LIBOR"]
    predictor_lst = []
    for L in range(0, len(_predictor_lst)+1):
        for subset in itertools.combinations(_predictor_lst, L):
            predictor_lst.append(list(subset))
    predictor_lst = predictor_lst[1:]
    label_lst = endog_lst
    
    model_lst = [DecisionTreeRegressor(random_state=0)]
    param_lst = [{'max_depth': [1,2,3,4]}]
    
    #r = Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    #
    #######################
    label_feature_pair = []
    for label in label_lst:
        for predictor in predictor_lst:
            label_feature_pair.append([label,predictor])
    
    print label_feature_pair
    ###############
    
    r_lst = Parallel(n_jobs=4)(delayed(rolling_grid_search_ML)
                           (model = model_lst[0], y = DataFrame(data[pair[0]]), X = data[pair[1]],
                                           group_size = 365, param_grid=param_lst[0], scoring = rmse, 
                                           crit = crit_min, window_size = 7, size_hyper_sel = 30)
                           for pair in label_feature_pair)
    
    print "\nRolling Forecasts Done!\n"
    
    print "Organizing Json......"
    new_r_lst = []
    for r,pair in zip(r_lst,label_feature_pair):
        new_r_lst.append({"actual": r['actual'],
                          "pred"  : r['pred'],
                          "score" : r['score'],
                          "params": r['params'],
                          "label" : pair[0],
                          "feature":pair[1]})     
    result_df = DataFrame(new_r_lst)[['label','feature','score']]
    result_df.to_csv(path_or_buf="CART_result.csv")
    # Output json to store results
    import json
    output = {"model": str(model_lst[0].__class__),
              "data" : new_r_lst}
    with open('json/ML_CART.json','w') as fp:
        json.dump(output,fp)