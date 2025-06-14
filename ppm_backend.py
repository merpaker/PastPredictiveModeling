#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmtp_backend.py

This file runs the past predictive modeling using the LightGBM model

"""

import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import statistics
import numpy as np
import shap
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance



""" 
TIME SERIES TRAIN TEST SPLIT
"""
def ts_split(data, forecast_year, yearid, window, earlyyear, lateyear):

    test_year = forecast_year - 1
    validation_year = forecast_year - 2
    
    # Ensure we only use data within the bounds passed to the function
    data0 = data[data[yearid] <= (lateyear)]
    data1 = data0[data0[yearid] >= (earlyyear)]


    # Expanding window: how far back to you want to go?
    data2 = data1[data1[yearid] >= (test_year - window)]
    

    # Training data: everything up to 3 years before test year
    df2_train = data2[data2[yearid] < (test_year - 2)]
    df2_validation = data2[data2[yearid]== validation_year]
    df2_test = data2[data2[yearid]== test_year]
    # Prediction year is test year + 1, AKA the forecast year
    df2_forecast = data2[data2[yearid]== forecast_year]
    forecast_year_inv = forecast_year*-1
    print("Predicted year: ", forecast_year_inv)
    
    return df2_train, df2_test, df2_validation, df2_forecast



"""
GET X COLS
"""
def get_xcols(data, excludevars, outcome):
    all_data = data.columns.tolist()
    all_data = [col for col in data.columns if col not in excludevars]
    x_cols = [var for var in all_data if var not in outcome]
    return x_cols


"""
BOOTSTRAP FUNCTION
"""
def bootstrap_predictions(X_train, y_train, X_forecast, skills, regions,outcome_logged, lgb_params, n_iterations=100):
  
    # Default hyperparameters if none provided
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',              
        'boosting_type': 'gbdt',
        'verbose': -1,
    }
    

    
    # Update default parameters with any user-provided parameters
    if lgb_params:
        default_params.update(lgb_params)
    
    # Initialize dictionaries to store bootstrap predictions
    bootstrap_predictions = {}
    
    for skill_type in skills.keys():
        bootstrap_predictions[skill_type] = {
            'overall': [],
            'regions': {region_name: [] for region_name in regions.keys()}
            }
    
    
    # Run bootstrap iterations
    for i in range(n_iterations):
        # Create a bootstrapped dataset
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_resampled, y_resampled = X_train.iloc[indices], y_train.iloc[indices]
        
        # Train an LGB model
        boot_model = lgb.LGBMRegressor(**default_params)
        boot_model.fit(X_resampled, y_resampled)
        
        # Make predictions for each skill type
        for skill_type, skill_column in skills.items():
            # Filter data for the current skill type
            filtered_skill_data = X_forecast[X_forecast[skill_column] == 1]
            
            if not filtered_skill_data.empty:
                try:
                    skill_preds = boot_model.predict(filtered_skill_data)
                    if outcome_logged == 1:
                        skill_preds = np.exp(skill_preds)
                    bootstrap_predictions[skill_type]['overall'].append(np.mean(skill_preds))
                except Exception as e:
                    print(f"Error during prediction for {skill_type}: {e}")
            
            # Make predictions for each region and skill combination
            for region_name, region_column in regions.items():
                filtered_skill_region_data = filtered_skill_data[filtered_skill_data[region_column] == 1]
                
                if not filtered_skill_region_data.empty:
                    try:
                        region_skill_preds = boot_model.predict(filtered_skill_region_data)
                        if outcome_logged == 1:
                            region_skill_preds = np.exp(region_skill_preds)
                        bootstrap_predictions[skill_type]['regions'][region_name].append(np.mean(region_skill_preds))
                    except Exception as e:
                        print(f"Error during prediction for {skill_type} in {region_name}: {e}")
    
    # Calculate bootstrap statistics
    bootstrap_stats = {}
    
    
    for skill_type in skills.keys():
        bootstrap_stats[skill_type] = {
            'overall': {
                'mean': None,
                'lower_ci': None,
                'upper_ci': None,
                'std_err': None
            },
            'regions': {
                region_name: {
                    'mean': None,
                    'lower_ci': None,
                    'upper_ci': None,
                    'std_err': None
                } for region_name in regions.keys()
            
            }
        }
    
    
    # Process overall predictions for each skill
    for skill_type in skills.keys():
        overall_preds = bootstrap_predictions[skill_type]['overall']
        if overall_preds:
            mean_pred = np.mean(overall_preds)
            std_pred = np.std(overall_preds)
            bootstrap_stats[skill_type]['overall'] = {
                'mean': mean_pred,
                'lower_ci': mean_pred - 1.96 * std_pred,
                'upper_ci': mean_pred + 1.96 * std_pred,
                'std_err': std_pred 
            }
        
        # Process region predictions for each skill
        for region_name in regions.keys():
            region_preds = bootstrap_predictions[skill_type]['regions'][region_name]
            if region_preds:
                mean_pred = np.mean(region_preds)
                std_pred = np.std(region_preds)
                bootstrap_stats[skill_type]['regions'][region_name] = {
                    'mean': mean_pred,
                    'lower_ci': mean_pred - 1.96 * std_pred,
                    'upper_ci': mean_pred + 1.96 * std_pred,
                    'std_err': std_pred 
                }
    
    return bootstrap_stats



"""
DEFINE MODEL ESTIMATION
"""
# 'data' must include CalendarYear
def model_estimate(forecast_year, window, data, yearid, outcome, outcome_logged, excludevars, earlyyear, lateyear, featureimportance, skills, regions, lgb_params, use_bootstrap, bootstrap_iterations):
    
    
    """
    GET X COLS
    """
    x_cols = get_xcols(data, excludevars, outcome)
    
    
    """
    TRAIN TEST SPLIT
    """
    train, test, validation, forecast = ts_split(data, forecast_year, yearid, window, earlyyear, lateyear)
    
    
    """
    CREATE X MATRIX AND Y
    """
    # Create X matrix and y for training
    X_train = train[x_cols].copy()
    y_train = train[outcome]
    
    # Create X matrix and y for validation
    X_validation = validation[x_cols].copy()
    y_validation = validation[outcome]
    
    # Create X matrix and y for testing
    X_test = test[x_cols].copy()
    y_test = test[outcome]

    # Create X matrix for predicting
    X_forecast = forecast[x_cols].copy()
    
    if X_test.empty:
        return [None, len(y_train), 0, len(X_forecast)] + [None] * (len(skills) + len(regions) * len(skills))

    if X_forecast.empty:
        return [None, len(y_train), len(y_test), 0] + [None] * (len(skills) + len(regions) * len(skills))
    

    
    """
    SCALING YEAR
   
    scaler = MinMaxScaler()
    X_train[yearid] = X_train[yearid].astype(float)
    X_train.loc[:, yearid] = scaler.fit_transform(X_train[[yearid]])
    X_test[yearid] = X_test[yearid].astype(float)
    X_test.loc[:, yearid] = scaler.transform(X_test[[yearid]])
    X_validation[yearid] = X_validation[yearid].astype(float)
    X_validation.loc[:, yearid] = scaler.transform(X_validation[[yearid]])
    X_forecast[yearid] = X_forecast[yearid].astype(float)
    X_forecast.loc[:, yearid] = scaler.transform(X_forecast[[yearid]])
"""

    """
    RUN LGB MODEL
    """
    
    # Default hyperparameters
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',              
        'boosting_type': 'gbdt',
        'verbose': -1,
    }

    # Update default parameters with any user-provided parameters
    if lgb_params:
        default_params.update(lgb_params)
    
    # Initialize LGB Regressor class with parameters
    lgb_model = lgb.LGBMRegressor(**default_params)

    # Fit model to training data
    lgb_model.fit(X_train, y_train)
    
    
    """
    EVALUATE FEATURE IMPORTANCE
    """
    
    if featureimportance == 1:
        
        fypos = forecast_year * -1
    
        # Create a SHAP explainer
        #explainer = shap.Explainer(lgb_model.booster_)
        #shap_values = explainer(X_train)
    
        #feature_important = np.abs(shap_values.values).mean(axis=0)
       # feature_names = X_train.columns 
        
        # Get feature importance based on gain
        #feature_important = lgb_model.booster_.feature_importance(importance_type='gain')
        #feature_names = X_train.columns
        
        if not X_validation.empty and X_validation.shape[0] > 0 and X_validation.shape[1] > 0:
        
            # Proceed only if validation data is non-empty
            if X_validation.shape[0] > 0 and X_validation.shape[1] > 0:
                result = permutation_importance(
                    lgb_model,                
                    X_validation,                    
                    y_validation,                    
                    n_repeats=10,             
                    random_state=64,
                    scoring='neg_mean_squared_error'  
                )
                
                # Extract and normalize importance scores
                feature_important = result.importances_mean
                importances = result.importances
                feature_names = X_validation.columns
                
                importances_5 = np.percentile(importances, 5, axis=1)
                importances_95 = np.percentile(importances, 95, axis=1)
                importances_mean = result.importances_mean
        
                total_importance = np.sum(np.abs(feature_important))
                feature_importance_pct = 100 * np.abs(feature_important) / total_importance    
                
                data_fi = pd.DataFrame({
                    "Feature": feature_names,
                    "Pct Score": feature_importance_pct,
                    "Importance Mean": importances_mean,
                    "Importance 5%": importances_5,
                    "Importance 95%": importances_95
                }).sort_values(by="Pct Score", ascending=False)
                
                # Add the year for tracking
                data_fi['Year'] = (forecast_year)*-1
                
                # Save the feature importance data
                data_fi.to_csv(f'feature_importance/feature_importance_{fypos}.csv', index=False)
            
    
    """
    SCORE MODEL
    """
    
    # Create out-of-sample score
    y_pred_lgb_out = lgb_model.predict(X_test)
    
    if len(y_test) > 0 and len(y_pred_lgb_out) > 0:
        mse_lgb_out = mean_squared_error(y_test, y_pred_lgb_out)
        rtscore_lgb = np.sqrt(mse_lgb_out)
        oos_obs = len(y_test)

    else:
        rtscore_lgb = None
        oos_obs = 0
    
    in_sample_obs =  len(y_train)
    forecast_obs = len(X_forecast)        


    """"
    CREATE FORECAST WITH LGB MODEL
    """
    

   # Initialize dictionaries to store wage predictions
    wage_predictions = {
        skill_type: {} for skill_type in skills.keys()
    }
    
    # Temporary variables for total wage predictions
    total_wage_predictions = {
        skill_type: None for skill_type in skills.keys()
    }
    
    
    # Nested loop for skills and regions
    for skill_type, skill_column in skills.items():
        # Filter data for the current skill type
        filtered_skill_data = X_forecast[X_forecast[skill_column] == 1]
        
        # Predict overall wage for the skill type
        try:
            y_pred_skill_forecasts = lgb_model.predict(filtered_skill_data)
            if outcome_logged == 1:
                y_pred_skill_forecasts = np.exp(y_pred_skill_forecasts)
            total_wage_predictions[skill_type] = statistics.mean(y_pred_skill_forecasts)
          
        except (statistics.StatisticsError, ValueError):
            total_wage_predictions[skill_type] = None
        
        # Nested loop for regions
        for region_name, region_column in regions.items():
            # Start with skill and region filtered data
            filtered_skill_region_data = filtered_skill_data[filtered_skill_data[region_column] == 1]
            
            # Skip if no data for this skill and region combination
            if filtered_skill_region_data.empty:
                wage_predictions[skill_type][region_name] = None
                continue
            
            # Predict wage for the specific skill and region
            try:
                y_pred_skill_region = lgb_model.predict(filtered_skill_region_data)
                if outcome_logged == 1:
                    y_pred_skill_region = np.exp(y_pred_skill_region)
                wage_predictions[skill_type][region_name] = statistics.mean(y_pred_skill_region)
            except (statistics.StatisticsError, ValueError):
                wage_predictions[skill_type][region_name] = None
         
                
    # Run bootstrap if enabled
    bootstrap_results = None
    if use_bootstrap:
        bootstrap_results = bootstrap_predictions(
            X_train, y_train, X_forecast, 
            skills, regions, outcome_logged,
            lgb_params=lgb_params,
            n_iterations=bootstrap_iterations,
        )
        
        bootstrap_df = pd.DataFrame()
    
        # Add overall predictions for each skill
        for skill_type, skill_stats in bootstrap_results.items():
            
            if skill_type == 'total':  # Skip 'total', already processed
                continue
            
            overall_stats = skill_stats.get('overall', {})
            if overall_stats and overall_stats.get('mean') is not None:
                row = {
                    "Year": forecast_year,
                    "Skill": skill_type,
                    "Region": "Overall",
                    "Lower_CI": overall_stats['lower_ci'],
                    "Upper_CI": overall_stats['upper_ci'],
                    "Std_Err": overall_stats['std_err']

                }
                
                bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame([row])], ignore_index=True)
            
            # Add region predictions for each skill
            for region_name, region_stats in skill_stats.get('regions', {}).items():
                if region_stats and region_stats.get('mean') is not None:
                    row = {
                        "Year": forecast_year,
                        "Skill": skill_type,
                        "Region": region_name,
                        "Lower_CI": region_stats['lower_ci'],
                        "Upper_CI": region_stats['upper_ci'],
                        "Std_Err": overall_stats['std_err']
                    }
                
                    bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame([row])], ignore_index=True)
        

    return_values = [ 
        rtscore_lgb, 
        in_sample_obs,
        oos_obs, 
        forecast_obs,
    ]

    
    # Add total predictions for each skill type
    skill_order = list(skills.keys())
    for skill in skill_order:
        return_values.append(total_wage_predictions.get(skill))
    
    # Add region-specific predictions for each skill and region
    for region in list(regions.keys()):
        for skill in skill_order:
            return_values.append(wage_predictions.get(skill, {}).get(region))
            
    # Calculate total expected number of return values
    expected_return_length = 5 + len(skills) + (len(regions) * len(skills))

    # Ensure the return values list has the expected length
    if len(return_values) < expected_return_length:
        # Pad with None values if needed
        return_values.extend([None] * (expected_return_length - len(return_values)))
    
    # Add bootstrap results if available
    if bootstrap_results is not None:
        return (tuple(return_values), bootstrap_results)
    else:
        return tuple(return_values)


    return tuple(return_values)




"""
DEFINE WALK FORWARD FUNCTION
"""

def walk_forward(start_year, end_year, years_extra, window, data, yearid, outcome, outcome_logged, excludevars, featureimportance, skills, regions, lgb_params, use_bootstrap, bootstrap_iterations):
    # Generate column names dynamically based on skills and regions
    column_names = [
        "Year", 
        "Out-Of-Sample (Test) RMSE", 
        "Training N",
        "Test N", 
        "Prediction N",
    ]
    
    # Add total predictions for each skill
    for skill in skills.keys():
        column_names.append(f"Overall Pred: {skill.title()}")
        if use_bootstrap:
            column_names.append(f"Overall {skill.title()} Lower CI")
            column_names.append(f"Overall {skill.title()} Upper CI")
            column_names.append(f"Overall {skill.title()} Std Err")  
    
    # Add region-specific predictions for each skill
    for region in regions.keys():
        for skill in skills.keys():
            column_names.append(f"{region.title()} Pred: {skill.title()}")
            if use_bootstrap:
                column_names.append(f"{region.title()} {skill.title()} Lower CI")
                column_names.append(f"{region.title()} {skill.title()} Upper CI")
                column_names.append(f"{region.title()} {skill.title()} Std Err")  
            
    
    # Create results DataFrame with dynamic columns
    results_df = pd.DataFrame(columns=column_names)
   
    
    i = start_year + years_extra
    while i <= end_year:
        
        model_result = model_estimate(
            i, window, data, yearid, outcome, outcome_logged, excludevars, 
            start_year, end_year, featureimportance,
            skills, regions, lgb_params,
            use_bootstrap, bootstrap_iterations
        )
        
        # Extract results based on bootstrap mode
        if use_bootstrap and isinstance(model_result, tuple) and len(model_result) == 2:
             results, bootstrap_stats = model_result
             
        else:
             results = model_result
             bootstrap_stats = None
        
        # Create row with dynamic column mapping
        row = {
            "Year": i,
            "Out-Of-Sample (Test) RMSE": results[0] if results[0] is not None else "",
            "Training N": results[1] if results[1] is not None else "",
            "Test N": results[2] if results[2] is not None else "",
            "Prediction N": results[3] if results[3] is not None else "",
        }
        
        total_pred_start_index =  4
        
        # Add total predictions for skills
        for j, skill in enumerate(skills.keys()):
            row[f"Overall Pred: {skill.title()}"] = results[total_pred_start_index + j] if results[total_pred_start_index + j] is not None else ""
        
            # Add bootstrap confidence intervals for overall predictions
            if use_bootstrap and bootstrap_stats is not None:
                skill_stats = bootstrap_stats.get(skill, {}).get('overall', {})
                row[f"Overall {skill.title()} Lower CI"] = skill_stats.get('lower_ci', "")
                row[f"Overall {skill.title()} Upper CI"] = skill_stats.get('upper_ci', "")
                row[f"Overall {skill.title()} Std Err"] = skill_stats.get('std_err', "")  
        
        
        # Add region-specific predictions
        region_pred_start_index = total_pred_start_index + len(skills)
        for region_idx, region in enumerate(regions.keys()):
            for skill_idx, skill in enumerate(skills.keys()):
                result_index = region_pred_start_index + (region_idx * len(skills)) + skill_idx
                row[f"{region.title()} Pred: {skill.title()}"] = results[result_index] if  results[result_index] is not None else ""
    
                # Add bootstrap confidence intervals for region-specific predictions
                if use_bootstrap and bootstrap_stats is not None:
                    region_stats = bootstrap_stats.get(skill, {}).get('regions', {}).get(region, {})
                    row[f"{region.title()} {skill.title()} Lower CI"] = region_stats.get('lower_ci', "")
                    row[f"{region.title()} {skill.title()} Upper CI"] = region_stats.get('upper_ci', "")   
                    row[f"{region.title()} {skill.title()} Std Err"] = region_stats.get('std_err', "")        
    
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if not pd.Series(row).isna().all():
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                
        
        i = i + 1
        
    

    
    return results_df

"""
DEFINE WALK BACKWARD FUNCTION
"""

# This function walks backward the years to estimate the score for each year, using walk forward function
def walk_backward(early_year, late_year, years_extra, window, data, yearid, outcome, outcome_logged, excludevars, featureimportance, skills, regions, lgb_params, use_bootstrap, bootstrap_iterations):
        
    print("\nOutcome variable: ", outcome)
    print("Year range: ", early_year, "to ", late_year)
    if featureimportance == 1:
        print("Feature importance will be computed for each year. Disable this option if the program runs too slowly.")
    if use_bootstrap == 1:
        print("Bootstrap standard errors will be computed for each year. Disable this option if the program runs too slowly.")
    print("___________________________________\n")
    data_copy = data.copy()
    start_year = late_year * -1
    end_year = early_year * -1
    data_copy['CalendarYearNeg'] = data_copy[yearid]*-1
    data_copy[yearid] = data_copy['CalendarYearNeg']
    results_df2 = walk_forward(start_year, end_year, years_extra, window, data_copy, yearid, outcome, outcome_logged, excludevars, featureimportance, skills, regions, lgb_params, use_bootstrap, bootstrap_iterations)
    results_df2['Year'] = results_df2['Year']*-1
 
    print("\n___________________________________\n")
    print("All LightGBM hyperparameters used:")
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',              
        'boosting_type': 'gbdt',
        'verbose': 0,
    }
    if lgb_params:
        default_params.update(lgb_params)
    
    # Create a temporary model to get all parameters
    temp_model = lgb.LGBMRegressor(**default_params)
    for param_name, param_value in sorted(temp_model.get_params().items()):
        print(f"{param_name}: {param_value}")

    print("\n___________________________________\n")
    print("-- Predicting Complete -- ")
    print("Predictions written to lgb.csv")
    if featureimportance == 1:
        print("Feature importance written to feature_importance folder\n")
    
    print("___________________________________\n")
    return results_df2



"""
RUN MODEL
"""

def run_pmtp(year_start, year_end, years_extra, window_size, path, filename, year_id, outcome_var, outcome_logged, exclude_vars, feature_importance, split1, split2, lgb_params, use_bootstrap, bootstrap_iterations):
    
    os.chdir(path)
    if feature_importance == 1:
        subdirectory_path = os.path.join(path, 'feature_importance')
        os.makedirs(subdirectory_path, exist_ok=True)

    df = pd.read_csv(filename,  low_memory = False)
    include_whole_sample = not split1 or len(split1) == 0 or not split2 or len(split2) == 0
    if include_whole_sample:
        df['Total'] = 1
        split1 = {
           'Total': 'Total'
        }
    results_1 = walk_backward(year_start, year_end, years_extra, window_size, df, year_id, outcome_var, outcome_logged, exclude_vars, feature_importance, split1, split2, lgb_params, use_bootstrap, bootstrap_iterations)
    results_1.to_csv('lgb.csv', index=False)



