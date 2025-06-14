#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
PAST PREDICTIVE MODELING ANALYSIS SCRIPT
PastPredictiveModeling.py

~ This version last updated: 14 June 2025 ~

===========================================================

PURPOSE:
This script is designed to run past predictive modeling (Paker, Stephenson, Wallis 202X)
using a Light Gradient Boosting (LGB) predictive machine learning model. 
It allows for users to configure the data, outcome variable, years of analysis,
included variables "features", and to make predictions across categorical subgroups.
Users may determine if they would like to generate feature importance results, if they
want bootstrap standard errors, and if they would like to use the default LightGBM 
hyperparameter tuning or their own tuning. 


PREREQUISITES:
1. Ensure you have the following Python libraries installed:
   - pandas
   - numpy
   - scikit-learn
   - lightgbm
   - shap
   - libomp

2. Prepare your data by cleaning and pre-processing a .csv file:
   - Contains a column with a four digit year
   - Contains a column of the outcome variable (log-transformed if desired)
   - Categorical variables should be encoded as dummy variables (0/1)
   - Continous variables except for year should be scaled ((x_i - mean(x))/sd(x))

CONFIGURATION INSTRUCTIONS:
Before running the script, carefully review and modify the USER-DEFINED INFORMATION 
section below. Each parameter is crucial for the correct execution of the model.

EXPECTED OUTPUTS:
  1. 'lgb.csv': Main results file with model fit score, predictions, and (optionally) bootstrap standard errors for all subgroups
  2. 'feature_importance/feature_importance_[year].csv': Detailed annual feature importance (if enabled)

TROUBLESHOOTING:
- Verify all file paths and column names exactly match your dataset
- Verify that all columns in your data that you do not want to use as features are specified in exclude_vars
- Verify that transformations of your outcome variable are excluded
- Check that categorical splits cover all relevant values in your data
- Ensure all required libraries are installed
- Remember that if using a logged outcome variable, you will need to exponentiate your final results
- Remember that the first predicted year will be 14 years before the 
  specified end year to guarantee at least 10 years of training data

RECOMMENDED WORKFLOW:
1. Prepare and clean your input data
2. Configure the parameters below
3. Run the script
4. Review output files and model performance
5. Refine configuration as necessary

"""



"""
===========================================================
USER-DEFINED INFORMATION
Edit this Section
===========================================================
"""

# 1. Change this path to the directory where your clean and pre-processed CSV file is located
# -- Use the FULL path to the directory, for example:
# ----- On Mac/Linux: '/Users/YourUsername/Documents/YourProjectFolder'
# ----- On Windows: 'C:\\Users\\YourUsername\\Documents\\YourProjectFolder'
path  = '/Users/merpaker/Documents/PPM Upload/'

# 2. Change this to the name of your CSV file
# -- Include the extension .csv
filename ='CleanWageData.csv'

# 3. Change this to the name of the column in your dataset representing the outcome variable 
# -- This is what you're trying to predict (analogous to dependent variable) 
# -- Note that if you put a logged value here, you will need to compute e^y of the final results
outcome_var = 'lnwage'

# 4. Is your outcome variables logged? Set this to 1 if so, or 0 otherwise
# -- The model will predict the logged outcome variable if this equals 1, but the output will be e^
# -- If bootstrap standard errors are estimated, these will be for the e^ predictions
outcome_logged = 1


# 5. Change this to the name of the column in your dataset representing the year variable
year_id = 'CalendarYear'


# 6. List here the names of any columns that should NOT be included in the model
# -- Only include columns that are dummy variables (0/1) or scaled continuous variables
# -- Column titles should be in single quotes, separated by commas
# -- All other columns not listed here will be features used in the model (excluding the outcome variable)
# -- You should not list the outcome variable here
exclude_vars = ['Ref_County',
                'Ref_Occ',
                'Ref_Status',
                'Ref_Craft',
                'Ref_Season',
                'Ref_Type',
                'Ref_Source',
                'Ref_Region',
                'Ref_WageType',
                'DayWage',
                'CalendarYearNeg']


# 7. Change these values to the start and end years you would like to use for the analysis
year_start = 1209
year_end = 1914

# 8. Determine how many years of additional data you require for the latest (most recent prediction)
# -- The default value is 14, meaning that the latest predicted year will be 14 years before year_end to ensure a minimum of 10 years of training data
# ----- E.g. if you set year_end = 1850, then your latest prediction will be 1836
# -- You should increase this value if your data are extremely scarce 
# -- It is recommended to ensure you always have min_data_in_leaf * num_leaves sample size in your training data
years_extra = 14

# 9. Change this feature importance flag to 1 to get feature importance results and 0 to skip
# -- Note that computing the feature importance results makes the analysis run more slowly
feature_importance = 0


# 10. Change this boostrap standard error flag to 1 to generate bootstrap standard errors and 0 to skip
# -- Note that computing bootstrap standard errors makes the analysis run more slowly
use_bootstrap = 0


# 11. Define subsets of the data for which you want different predictions
# -- You can define two layers of categorical variables for which you want predictions
# -- The first layer, split 1, will generate a prediction for each defined value of the split
# -- The second layer, split 2, will generate a prediction for each combination of split 1 and split 2
# -- E.g. If you define split1 to be skilled vs unskilled, and split2 to be regions, then 
# you will get a prediction for skilled, unskilled, region 1 skilled, region 1 unskilled, region 2 skilled, region 2 unskilled, etc.
# -- If you want to calculate for the whole dataset with no subsamples, leave split 1 and split 2 blank

# These dictionaries map your descriptive labels to specific column names
# Your label goes on the left of the colon and will be in the output
# The column name goes on the right of the colon
# You must include a line for EACH VALUE of the category you want split on (which should be captured by a different 0/1 variable in the data)

# Edit the keys and values to match your specific dataset categories 

split1 = {
   'laborer': 'Type_3',
   'craftsman': 'Type_2'
}

   
split2 = {
   'london': 'Region_1',
   'north': 'Region_3',
   'midlands': 'Region_2',
   'south east': 'Region_4',
   'south west': 'Region_5'
}


# 12. (Advanced) If you would like to change the Light GBM hyperparameters, save new parameters 
# Otherwise, leave this set to none and the defauly LightGBM parameters will be used


#lgb_params = None

# Params for Clark (2005) lightGBM
lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 10,
        'num_leaves': 31,
        'learning_rate': 0.2,
        'n_estimators': 500,
        'min_data_in_leaf': 10,
        'lambda_l1': 1,
        'feature_fraction': 0.9,
        'verbosity': -1
}



"""
===========================================================
RUNNING THE MODEL
Do not edit this section
===========================================================
"""

import ppm_backend
ppm_backend.run_pmtp(year_start, year_end, years_extra, 2000, path, filename, year_id, outcome_var, outcome_logged, exclude_vars, feature_importance, split1, split2, lgb_params, use_bootstrap, 100)



