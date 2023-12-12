#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTS ALL LIBRARIES
import pandas as pds
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



from sklearn.impute import SimpleImputer


#IMPORT DATASET
TheData=pds.read_excel(r'E:\BIGDATA\PCOS_data.xlsx')

print(TheData.columns)


# In[2]:


imputer = SimpleImputer(strategy='median')


# In[3]:


duplicates = TheData[TheData.duplicated(keep = False)]
print(duplicates)

# This will remove all duplicate rows
TheData.drop_duplicates(inplace=True)


# In[9]:


#DESCRIPTIVE ANALYSIS ON THE DATASET
TheData.describe()


# In[22]:


X=TheData.drop(columns=['Sl. No','Patient File No.', 'Fast food (Y/N)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)', 'PCOS (Y/N)', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'AMH(ng/mL)', 'Pimples(Y/N)'])
X = X.dropna()
y=TheData['PCOS (Y/N)']
X_imputed = imputer.fit_transform(X)

print(X_imputed.shape)
print(y.shape)

import numpy as np
# Identify the index of the row causing the mismatch
index_to_drop = 500 # Your index value causing the mismatch

# Drop the corresponding row in X_imputed
#X_imputed = np.delete(X_imputed, index_to_drop, axis=0)

# Drop the corresponding entry in y
y = np.delete(y, index_to_drop)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_for_train,x_for_test,y_for_train,y_for_test=train_test_split(X_imputed,y,test_size=0.2)

#Create a Decision Tree, Logistic Regression, Support Vector Machine and Random Forest Classifiers
Decision_Tree_Model=DecisionTreeClassifier()
Logistic_Regression_Model = LogisticRegression(max_iter=10000)
#Logistic_Regression_Model=LogisticRegression()
Support_Vector_Machine_Model=svm.SVC(kernel='linear')
Random_Forest_Model=RandomForestClassifier(n_estimators=100)

#TRAIN THE MODEL USING THE TRAINING SETS
Decision_Tree_Model.fit(X_for_train,y_for_train)
Logistic_Regression_Model.fit(X_for_train,y_for_train)
Support_Vector_Machine_Model.fit(X_for_train,y_for_train)
Random_Forest_Model.fit(X_for_train,y_for_train)

#PREDICT THE MODEL
DT_prediction=Decision_Tree_Model.predict(x_for_test)
LR_prediction=Logistic_Regression_Model.predict(x_for_test)
SVM_prediction=Support_Vector_Machine_Model.predict(x_for_test)
RF_prediction=Random_Forest_Model.predict(x_for_test)

# Calculate accuracy scores
DT_score = accuracy_score(y_for_test, DT_prediction)
LR_score = accuracy_score(y_for_test, LR_prediction)
SVM_score = accuracy_score(y_for_test, SVM_prediction)
RF_score = accuracy_score(y_for_test, RF_prediction)

#DISPLAY ACCURACY
print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", LR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")

from joblib import dump

# Assuming X_imputed and y have been prepared and X_imputed is already imputed

# Creating and training the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_imputed, y)

# Saving the trained model to a file
dump(model, 'E:\BIGDATA\ThePCOS_model.joblib')


# In[ ]:





# In[ ]:




