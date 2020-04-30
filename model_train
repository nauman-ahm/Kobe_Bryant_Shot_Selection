# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

import os
print(os.listdir("../input"))

## Reading data into environment
data = pd.read_csv("../input/data.csv")
#Checking NA's in dataset
data.isnull().sum()

# Shot ID's for submission data
shot_ids = (data[data.shot_made_flag.isnull() == True]).shot_id

#Feature Engineering
## Adding a year column
data['yr'] = data.game_date[:4]
## Adding column representing if Lakers are visitng another Team
data['is_away'] = data.matchup.str.contains('@').astype(int)
## Adding a column for total time remaining
data['total_time'] = data.seconds_remaining + (60*data.minutes_remaining)
## Log transform of total_time
data['total_time_log'] = np.log(data.total_time + 1)

#Dropping unwanted and redundant features
data = data.drop(['season', 'game_id','lat','lon',
'team_id','team_name','game_date','matchup','shot_id','shot_zone_range'], axis = 1)

#Filtering total_time for only 4th quarter
data = data[(data.shot_type == "2PT Field Goal") & (data.total_time <100)]

#One Hot Encoding
data = pd.get_dummies(data, columns = ['yr','action_type','combined_shot_type','shot_distance','minutes_remaining','period','playoffs','shot_type','opponent', 'shot_zone_area','shot_zone_basic'])

# Seperating submission data out
submission_data = data[data.shot_made_flag.isnull() == True]
data = data[data.shot_made_flag.isnull() == False]

submission_data_X = submission_data.drop(['shot_made_flag'], axis =1)
submission_data_y = submission_data.shot_made_flag

data_X = data.drop(['shot_made_flag'], axis =1 )
data_y = data.shot_made_flag

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(data_X,data_y, random_state = 41, test_size = 0.3)

# Modelling with Logit
model_logit = LogisticRegression() # Hyper params found through Grid Search shown below
param_grid_logit = {'penalty':['l1','l2'],'C': np.arange(0.025,1.0, 0.025)}
GS_logit = GridSearchCV(model_logit, param_grid_logit, cv = 5, scoring = 'roc_auc')
GS_logit.fit(X_train, y_train)
roc_auc_score(y_true = y_test, y_score = GS_logit.predict_proba(X_test)[:,1])
log_loss(y_true = y_test, y_pred = GS_logit.predict_proba(X_test)[:,1])


# Modelling with XGBoost
#model_XGB = XGBClassifier(objective="binary:logistic", eval_metric = "logloss")
#param_grid_XGB = {'max_depth':np.arange(5,10,1), 'reg_lambda':np.arange(0,2,0.25), 'reg_alpha':np.arange(0,2,0.25)}
#GS_XGB = GridSearchCV(model_XGB, param_grid_XGB)
#GS_XGB.fit(X_train, y_train)
#log_loss(y_true = y_test, y_pred = GS_XGB.predict_proba(X_test))
#roc_auc_score(y_true = y_test, y_score = GS_XGB.predict_proba(X_test))


# Submission
#submission= pd.DataFrame({'shot_id':shot_ids, 'shot_made_flag': model.predict(submission_data_X)})
#submission.to_csv(index = False, path_or_buf ='submission.csv')
