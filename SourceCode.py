# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 00:26:48 2016

@author: Vybhav

    The data is stored in same folder and will be using for preprocessing and after preprocessing
wil be stored as clean_train,Clean_Holdout data sets.


"""



import os
import seaborn as sns    
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rep.estimators import XGBoostRegressor
from sklearn import linear_model
import time

os.chdir(r'G:\State Farm Data Science WORK SAMPLE')


    
def data_preprocessing(SF_Train):    
    #######################Checking the missing values row wise
    for i ,j in zip(SF_Train.count(axis=1),SF_Train.index):
        if i <=16:
            SF_Train.drop([j],inplace=True)	
    ###################### Removing the special characters        
    for i in ('X4','X5','X6'):
        SF_Train[i]=SF_Train[i].map(lambda x:str(x).replace('$','')) 
        SF_Train[i]=SF_Train[i].map(lambda x:str(x).replace(',',''))         
        SF_Train[i]=SF_Train[i].astype(int)
        
    for i in ('X1','X30'):
        SF_Train[i]=SF_Train[i].map(lambda x:str(x).replace('%','')) 
        SF_Train[i]=SF_Train[i].map(lambda x: round(float(x)/100,4))
    ####################### checking for the missing values 
    if np.mean(SF_Train.isnull().sum())>0:    
        Missing_data=pd.DataFrame(columns=['feature','f_type','m_count','m_per','del'])
        j=0
        for i in SF_Train.columns:
            if SF_Train[i].isnull().sum():
                Missing_data.loc[j,'feature']=i
                Missing_data.loc[j,'f_type']= SF_Train[i].dtype
                Missing_data.loc[j,'m_count']= SF_Train[i].isnull().sum()
                Missing_data.loc[j,'m_per']=SF_Train[i].isnull().sum()/float(SF_Train.shape[0])
                Missing_data.loc[j,'levels']=len(SF_Train[i].value_counts().index)
                Missing_data.loc[j,'del']=((SF_Train[i].isnull().sum()/float(SF_Train.shape[0]))>0.5).astype(int)
                j=j+1        
    ################ Missing values imputation 
    SF_Train['X9']=SF_Train['X9'].fillna(SF_Train['X9'].mode()[0])
    X12_mode=SF_Train['X12'].mode()[0] 
    SF_Train['X12']=SF_Train['X12'].fillna(X12_mode)
    SF_Train['X13']=SF_Train['X13'].fillna(SF_Train['X13'].median())
    SF_Train['X30']=SF_Train['X30'].fillna(SF_Train['X30'].median())
    ### handling <1 year,10+ years,

    SF_Train['X11'] = SF_Train['X11'].map(lambda x: '0.5 years' if x == '< 1 year' else x)
    SF_Train['X11'] = SF_Train['X11'].map(lambda x: '10 years' if x == '10+ years' else x)
    SF_Train['X11'] = SF_Train['X11'].map(lambda x: '0 years' if x == 'n/a' else x)
    SF_Train['X11'] = SF_Train['X11'].map(lambda x: float(x.strip(' years')))
    #SF_Train.columns
    for i in ('X2','X3','X4','X6','X8','X10','X15','X16','X18','X19','X20','X23','X25','X26'):
        del SF_Train[i]
    categorical_features=[]    
    for i in SF_Train.columns:
        if SF_Train[i].dtype=='object':
            if i != 'label':
                categorical_features.append(i)
    SF_Train = pd.get_dummies(SF_Train, columns=categorical_features)
    train_data=SF_Train[SF_Train['label']=='train']
    test_data=SF_Train[SF_Train['label']=='test']
        
    del train_data['label']
    del test_data['label']
    train_data= train_data[train_data['X1'].notnull()]        
    train_data.to_csv('Clean_train.csv')
    test_data.to_csv('Clean_Holdout.csv')
    return(train_data,test_data)
    
####################### helper functions    

def RMSE(clf,X_test,y_test):
    return(mean_squared_error(y_test,clf.predict(X_test) )**0.5)
    
    
######################## Data modeling
def data_modelling(model,train_cl,test_cl):    
    y_train=train_cl.X1.values
    train_cl = train_cl.drop('X1',axis=1)
    X_train=train_cl.values
    #del test_cl['X1']
    ######## Cross validation code-- uncomment to run 
    #X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42) 
    #Ridge = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],cv=10,normalize=True)
    
#    print "Ridge cv"
#    print "R-squared for Train: %.6f" %Ridge.score(X_train, y_train) 
#    print "R-squared for Test: %.6f" %Ridge.score(X_test, y_test)
#    print("Residual sum of squares: %.6f" %np.sqrt( np.mean((Ridge.predict(X_test) -y_test) ** 2)))
    #print np.sqrt(mean_squared_error(y_test,(Ridge.predict(X_test))))    
#    Train	: 0.826672
#    Validation 	: 0.831106
#    RMSE 	: 0.018050

#    rf2=RandomForestRegressor(n_estimators=200,bootstrap=True,oob_score = True, min_samples_leaf=1, min_samples_split=1,random_state=55)    
#    scores = cross_validation.cross_val_score(rf2,X_train, y_train, cv=5,scoring=RMSE)
#
#    scores.mean(), scores.std()
#    (0.012511905180054956, 0.00010757453384678779)
    
    model.fit(X_train,y_train)
    predictions=model.predict(test_cl)    
    #print len(predictions)
    #print np.NAN(predictions)
    return(predictions)    


########################### main ()#########################


def main():
    parent_dir = os.path.dirname(os.path.realpath('__file__'))
    train_file = parent_dir + r'/Data for Cleaning & Modeling.csv'
    test_file = parent_dir + r'/Holdout for Testing.csv'
    #output_file = parent_dir + r'/Holdout for Testing.csv'
    ########################### inputing the data 
    train_data=pd.read_csv(train_file,low_memory=False)
    test_data=pd.read_csv(test_file,low_memory=False)    
    train_data['label']='train'
    test_data['label']='test'
    #######################Concate both train, test data sets
    full_data = pd.concat([train_data,test_data],axis=0)
    
    train_clean,test_clean = data_preprocessing(full_data)
    del test_clean['X1']    
#    print train_clean.shape
#    print test_clean.shape
    save_predictions=pd.DataFrame(columns=(['Ridge','Randomforest']))


    #########################RidgeCV #####################
    ridge= linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],cv=10,normalize=True)    
    save_predictions['Ridge']= data_modelling(ridge,train_clean,test_clean)
    #print save_predictions['Ridge'].head()
    #print len(save_predictions['Ridge'])
    #######################Random forest #################
    rf=RandomForestRegressor(n_estimators=200,oob_score = True, 
                             random_state=55)    
    save_predictions['Randomforest']= data_modelling(rf,train_clean,test_clean)
    #print save_predictions['Randomforest'].head()
    #print len(save_predictions['Randomforest'])
    
    print save_predictions.shape
    save_predictions.to_csv('Results from vybhavreddy_kc.csv')



if __name__ == '__main__':
    main()