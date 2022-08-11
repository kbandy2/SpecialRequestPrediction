# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:24:03 2022

@author: kyleb
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

converter={"is_canceled":str,
           "is_repeated_guest":str,
           "agent":str,
           "company":str,
           }


df = pd.read_csv(r"hotel_bookings.csv", converters=converter)

df.head()

df.tail()

df.dtypes

df.shape 

#Remove columns that won't be used in classifier
#Year and day of month is irrelevant, but month of year has seasonality and may be of interest
#The reservation status fields are purely informational

df = df.drop(["arrival_date_year", "arrival_date_day_of_month", 
              "reservation_status", "reservation_status_date", 
              "arrival_date_week_number", "country", "agent", "company",
              "arrival_date_month", "market_segment", "distribution_channel"], axis=1)

#Deal with NaN's
df['children'] = df['children'].fillna(0)


#Get rid of the ADR outlier
df = df[df["adr"] <1000]

#Split the response from the predictors
X = df.drop(['total_of_special_requests'], axis=1)

y = df['total_of_special_requests']

y = np.where(y >=2, "2+", y)

y = [str(i) for i in y]


val, count = np.unique(y, return_counts=True)
sns.set_style("white")
plt.bar(x=range(len(val)), height= count, edgecolor="black", alpha=.9)
plt.title(r"Distribution of Total Number of Special Requests")
plt.xticks([0,1,2], ["0", "1", "2+"])
plt.xlabel(r"Number Special Requests")
plt.ylabel(r"Count of Reservations")
plt.tight_layout()
plt.show()


#Plot out some of the variables
sns.boxplot(y=df["adr"], x=y)
plt.title(r"Special Requests by ADR")
plt.ylabel(r"ADR")
plt.xlabel(r"Special Requests")


sns.boxplot(y=df["lead_time"], x=y)
plt.title(r"Special Requests by Lead Time")
plt.ylabel(r"Lead Time")
plt.xlabel(r"Special Requests")



#Encode the categorical variables
X = pd.get_dummies(X)

feature_names = X.columns


#Scale the data for use in ML models
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)


#Need to deal with the fact that the classes are so imbalanced
#Upsample the smaller classes using SMOTE (Synthetic Minority Over-Sampling Technique)

sm = SMOTE(random_state = 123)
X_train, y_train = sm.fit_resample(X_train, y_train)


#Make sure that SMOTE did what it was supposed to do
val, count = np.unique(y_train, return_counts=True)
sns.set_style("white")
plt.bar(x=range(len(val)), height= count, edgecolor="black", alpha=.9)
plt.title(r"SMOTE Distribution of Total Number of Special Requests")
plt.xticks([0,1,2], ["0", "1", "2+"])
plt.xlabel(r"Number Special Requests")
plt.ylabel(r"Count of Reservations")
plt.show()




#Classification Models
    
#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_score = 0
best_k = 0
for i in np.arange(1,11):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1).fit(X_train, y_train)
    this_score = knn.score(X_test, y_test)
    print(i,":",this_score)
    if this_score > knn_score:
        knn_score = this_score
        best_k = i
        
knn = KNeighborsClassifier(n_neighbors=345, n_jobs=-1).fit(X_train, y_train)
knn.fit(X_test, y_test)
knn_score = knn.score(X_test, y_test)
print(knn_score)


#Logistic Regression

from sklearn.linear_model import LogisticRegressionCV

lr = LogisticRegressionCV(n_jobs=-1)
lr.fit(X_train, y_train)

lr_score = lr.score(X_test, y_test)
print(lr_score)



#Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

for i in np.arange(.1,1.1,.1):

    nb = MultinomialNB(alpha=i, class_prior = [.58,.27,.15])
    nb.fit(X_train, y_train)
    
    nb_score = nb.score(X_test, y_test)
    
    print(i,":",nb_score)


#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

dtc_score = dtc.score(X_test, y_test)
print(dtc_score)

dtc_imp = dtc.feature_importances_
print(dtc_imp)

dtc_imp_df = pd.DataFrame({"Feature Name":feature_names, "DTC Importance":dtc_imp}).sort_values(by="DTC Importance", ascending=False)


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_score =rf.score(X_test, y_test)

print(rf_score)

rf_imp = rf.feature_importances_
print(rf_imp)

rf_imp_df = pd.DataFrame({"Feature Name":feature_names, "RF Importance":rf_imp}).sort_values(by="RF Importance", ascending=False)


#Let's use this for variable selection since it performed well; train the model on the first five.

features = list(rf_imp_df.index[:5])

X_test, X_train = X_test[:,features], X_train[:,features]


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [80, 90, 100, 110],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 200, 300]
}

rf_gs = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf_gs, param_grid = param_grid,
                           cv = 3, n_jobs = -1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_


rf_gs = RandomForestClassifier(max_depth=100, min_samples_leaf = 1, min_samples_split=2, n_estimators = 200)
rf_gs.fit(X_train, y_train)

rf_gs_score = rf_gs.score(X_test, y_test)

print(rf_gs_score)

rf_gs_features = rf_gs.feature_importances_


#Plot out the feature importances
rf_gs_features = pd.DataFrame({"Feature Name":feature_names, "RF Importance":rf_gs_features}).sort_values(by="RF Importance", ascending=False)

sns.barplot(y='Feature Name', x='RF Importance', data=rf_gs_features[0:5])
plt.title(r"Feature Importance After Tuning")
plt.tight_layout()
plt.ylabel(None)
plt.savefig(r"rf_imp.png", dpi=800)

rf_gs_top = RandomForestClassifier(max_depth=100, min_samples_leaf = 1, min_samples_split=2, n_estimators = 200)

X_fs_train = X_train[['lead_time','adr','stays_in_week_nights','stays_in_weekend_nights','adults']]
X_fs_test = X_test[['lead_time','adr','stays_in_week_nights','stays_in_weekend_nights','adults']]


rf_fs = RandomForestClassifier()
rf_fs.fit(X_train, y_train)

rf_score_fs =rf_fs.score(X_test, y_test)

print(rf_score_fs)

rf_imp_fs = rf_fs.feature_importances_
print(rf_imp_fs)

rf_imp_df_fs = pd.DataFrame({"Feature Name":feature_names[features], "RF Importance":rf_imp_fs}).sort_values(by="RF Importance", ascending=False)


#Fin




























