# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:28:38 2022

@author: DELL
"""

import numpy as np
import pandas as pd

#importing required packages
import seaborn as sns 



import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("D:\dataset_science\data ds.xlsx")
df.columns
df.isna().sum()
df.info()
df.isnull().sum



# create Emotional Difficulties dataset
df.drop(columns=['time_fall_asleep','time_wake_up_TODAY','from_house_esly_walk_smwhr_play','have_garden','somewhere_athome_where_you_relax','Your Family','From_your_house_easily_walk_to_park','time_brushteeth','have_time_for_play','Your School','Your Friends ','Your_Appearance','Your Health',
                          'Your Life','safe_feel_playing_your_area','days_watch_TV_play_online_games','Age','going_to_school','Your Friends','how_many_days_do_exercise','touch_with_your_friends','how_are_you_keeping_in_touch'], inplace=True)

df.isna().sum() 
df=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
df.columns
columns = df.columns
df_outputs = df.iloc[:, 26:32]

# Behaviour Diffiulties
behaviour_outputs = df_outputs.iloc[:, :6]

behaviour_outputs.loc[:,'BD_scores'] = behaviour_outputs.sum(axis=1)
df['BD_scores'] = behaviour_outputs['BD_scores']



behaviour_outputs['BD_scores']=behaviour_outputs['BD_scores'].replace([0,1,2,3,4,5],'expected')
behaviour_outputs['BD_scores']=behaviour_outputs['BD_scores'].replace([6],'borderline')
behaviour_outputs['BD_scores']=behaviour_outputs['BD_scores'].replace([7,8,9,10,11,12,13,14,15],'clinically significant difficulties')
behaviour_outputs['BD_scores'].unique()

duplicate = df.duplicated()
sum(duplicate)
df =df.drop_duplicates()
df

from statsmodels.stats.outliers_influence import variance_inflation_factor

df_X = df.drop(columns = ['BD_scores'])
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["BD_scores"] = df_X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df_X.values, i)
 for i in range(len(df_X.columns))]

print(vif_data)

## multinominal regression before feature selection##

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = behaviour_outputs.drop(['BD_scores'],axis=1)  #independent
y = behaviour_outputs["BD_scores"]   #target
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2)

lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

model = accuracy_score(y_train, lr.predict(X_train))
model = accuracy_score(y_test, y_pred)

print(f"Training Accuracy of Logistic Regression Model is {model}")
#Training Accuracy of Logistic Regression Model is 1
print(f"Test Accuracy of Logistic Regression Model is {model}")
#Test Accuracy of Logistic Regression Model is 0.9441798941798942


#random forest
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = 'gini', max_depth = 5, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 5, n_estimators = 120)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)

rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy of Random Forest Model is {rand_clf_train_acc}")
#Training Accuracy of Random Forest Model is 0.9285294117647058
print(f"Test Accuracy of Random Forest Model is {rand_clf_test_acc}")
#Test Accuracy of Random Forest Model is 0.9485294117647058


import pickle
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

new_data = np.array([[4,7,4,3,2,2]])
lr.predict(new_data)


behaviour_outputs.I_get_very_angry




