def rolling_grid_search_ML (model, X, y, group_size, param_grid, scoring, crit, 
                            window_size, size_hyper_sel):
    # Import necessary libraries
    import collections
    from sklearn.model_selection import TimeSeriesSplit,ParameterGrid
    import pandas as pd
    import numpy as np
    
    def error_check(model, X, y, group_size, param_grid, scoring, crit, 
                    window_size, size_hyper_sel):
        if (group_size < size_hyper_sel):
            print "Error!"
            return -1
    def header_output():
        # Output Header and Information
        len_X = len(X)
        print "Rolling Grid Search for Machine Learning Models"
        print "By John Tsang"
        print "-------------------------------------------------"
        print "Model             :\n",model
        print "Length of Data Set:",len(X)
        print "Group size        :",group_size
        print "Labels (y)        :",y.columns.tolist()
        print "Features (X)      :",X.columns.tolist()
        print "Window Size       :",window_size
        print "Hyperparameter    :",size_hyper_sel
        print "Selection Size\n"
        return 
    
    def group_data():
        group_X_lst = []
        group_y_lst = []
        num_of_groups = len_X / group_size
        counter = 0
        for group_num in range(1,num_of_groups+1):
            counter += 1
            start = (group_num - 1) * group_size
            end   = (group_num    ) * group_size
            group_X_lst.append(X.iloc[start:end].copy())
            group_y_lst.append(y.iloc[start:end].copy())
            print "[",counter,"] Start: ",start," End:",end
        if (len_X % group_size != 0):
            counter += 1
            start = num_of_groups * group_size
            end = len_X
            group_X_lst.append(X.iloc[start:end].copy())
            group_y_lst.append(y.iloc[start:end].copy())
            print "[",counter,"] Start: ",start," End:",end
            num_of_groups += 1
        print "Number of groups  :",num_of_groups,"\n"
        Groups = collections.namedtuple('Groups', ["X","y",'n_groups'])
        g = Groups(X=group_X_lst, y = group_y_lst, n_groups = num_of_groups)
        return g    
    
    def grid_rolling_hyperparam_sel(model,X,y,param_grid,
                                    window_size,scoring,crit):
        len_X = len(X)
        #print "   in function len:",len_X
        # Convert X and y to data structure that can be inputs 
        # for scikit learn models
        X = X.values
        y = y.values
        # Generate cv for one-step-ahead forecast
        tscv_rolling = TimeSeriesSplit(n_splits=len_X-window_size, 
                                       max_train_size = window_size)
        # Initialize lists to store params and scores
        param_lst = []
        score_lst = []
        
        # Generate all combinations of hyperparameters for Grid Search
        param_grid = ParameterGrid(param_grid)
        # Grid Search using the first []
        for params in param_grid:
            actual_lst = []
            pred_lst   = []
            for train_index, test_index in tscv_rolling.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = model.set_params(**params)
                model.fit(X_train,y_train)
                actual_lst.append(y_test)
                y_pred = model.predict(X_test)
                pred_lst.append(y_pred)
            score_lst.append(scoring(actual_lst, pred_lst))
            param_lst.append(params)
        # Find optimal param for this X,y pair
        rt_tuple = crit(score_lst)
        # Store
        rt = dict()
        rt["best_params"] = param_lst[rt_tuple[0]]
        rt["optimal_score"] = rt_tuple[1]
        return rt
    def hyperparam_sel_wrap(group_X, group_y, num_of_groups,len_hyp_sel):
        #len_hyp_sel = size_hyper_sel
        # Tune hyperparameters
        print "Hyperparameter Tuning......"
        counter = 0
        best_params_lst = []
        tscv_rolling = TimeSeriesSplit(n_splits=len_X-window_size, max_train_size = window_size)    
        for group_X, group_y in zip(group_X_lst,group_y_lst):
            counter += 1
            if (num_of_groups != counter):
                # Select samples for tuning
                group_X = group_X[0:len_hyp_sel].copy()
                group_y = group_y[0:len_hyp_sel].copy()
                print "    [",counter,"] Length:",len(group_X)
                #print "Range: ",0,"to",len_hyp_sel
                # Hyperparameter tuning implementation
                rt = grid_rolling_hyperparam_sel(model = model, X = group_X, 
                                                 y = group_y, param_grid = param_grid, 
                                                 window_size = window_size, scoring = scoring, 
                                                 crit = crit)
                #print "[",counter,"] Length:",len(group_X),rt
                best_params_lst.append(rt["best_params"])
            else :
                if (len(group_X) > len_hyp_sel):
                    group_X = group_X[0:len_hyp_sel].copy()
                    group_y = group_y[0:len_hyp_sel].copy()
                    print "    [",counter,"] Length:",len(group_X)
                    rt = grid_rolling_hyperparam_sel(model = model, X = group_X, 
                                                     y = group_y, param_grid = param_grid, 
                                                     window_size = window_size, scoring = scoring, 
                                                     crit = crit)
                    best_params_lst.append(rt["best_params"])
                else:
                    print "    [",counter,"] Length:",len(group_X)," Use previous params"
                    best_params_lst.append(best_params_lst[-1])
        return best_params_lst
    
    def rolling_forecast(model, best_params_lst, group_X_lst, group_y_lst, window_size, size_hyper_sel, number_of_groups):
        print "\n"
        print "Rolling Forecast......"
        #print "    window size =",window_size
        pred_lst = []
        actual_lst = []
        counter = 0
        for group_X, group_y, best_params in zip(group_X_lst,group_y_lst, best_params_lst):
            counter += 1
            print "    Predicting group [",counter,"]"
            # Rolling Forecast using best_params
            model = model.set_params(**best_params)
            # Convert X and y to data structure that can be inputs 
            # for scikit learn models
            if (counter != number_of_groups):
                group_X = group_X[size_hyper_sel:].values
                group_y = group_y[size_hyper_sel:].values
            else:
                if (len(group_X) > size_hyper_sel):
                    group_X = group_X[size_hyper_sel:].values
                    group_y = group_y[size_hyper_sel:].values
                else:
                    group_X = group_X
                    group_y = group_y
            print "    Index: ",size_hyper_sel," Length:",len(group_X)
            # Generate cv for one-step-ahead forecast
            len_X = len(group_X)
            tscv_rolling = TimeSeriesSplit(n_splits=len_X-window_size, max_train_size = window_size)
            for train_index, test_index in tscv_rolling.split(group_X):
                #print "A"
                # Divide samples for training and testing
                #print "    Train:",train_index,"  Test:",test_index
                X_train, X_test = group_X[train_index], group_X[test_index]
                y_train, y_test = group_y[train_index], group_y[test_index]
                # Fit the model with training data
                model.fit(X_train,y_train)
                # Store the actual value
                actual_lst.append(y_test.item(0))
                # Store predicted values
                y_pred = model.predict(X_test)
                #print "      Predicted Value: ", y_pred, "  Type:",type(y_pred), " Item:",y_pred.item(0)
                #print "      Actual    Value: ", y_test, "  Type:",type(y_test), " Item:",y_test.item(0)
                pred_lst.append(y_pred.item(0))
        # Compute scores for this forecast
        score = scoring(actual_lst, pred_lst)
        print "\nScore:",score,"\n"
        # Pack results into a named tuple as return of the function
        # score:  the score of this forecast
        # params: list of hyperparameters used in this forecast
        # actual: list of actual exchange rates
        # pred  : list of predcited values
        ###################################################
        #RollingGrid = collections.namedtuple('RollingGrid', ['score', 'params', 'actual', 'pred'])
        #p = RollingGrid(score=score, params = best_params_lst, actual = actual_lst, pred = pred_lst)
        #p = (score, best_params_lst, actual_lst, pred_lst)
        p = dict()
        p['score']  = score
        p['params'] = best_params_lst
        p['actual'] = actual_lst
        p['pred']   = pred_lst
        return p
    # Main Program
    er = error_check(model, X, y, group_size, param_grid, scoring, crit, 
                     window_size, size_hyper_sel)
    if (er == -1):
        return -1
    header_output()
    len_X = len(X)
    # Group data 
    group = group_data()
    group_X_lst = group.X
    group_y_lst = group.y
    num_of_groups = group.n_groups
    # Hyperparameter Selection for each group
    len_hyp_sel = size_hyper_sel
    best_params_lst = hyperparam_sel_wrap(group_X_lst, group_y_lst, num_of_groups, size_hyper_sel)
    
    # Rolling forecast
    rt = rolling_forecast(model = model, best_params_lst = best_params_lst, 
                          group_X_lst = group_X_lst, group_y_lst = group_y_lst, 
                          window_size = window_size, size_hyper_sel = size_hyper_sel,
                          number_of_groups = num_of_groups)
    #rt = rt.score
    return rt