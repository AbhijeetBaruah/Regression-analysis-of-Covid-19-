# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 07:58:05 2020

@author: Abhijeet
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import matplotlib.pyplot as plt
#regression models import
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
#import metrics models
from sklearn.metrics import mean_absolute_error

import datetime

def main():
    st.title("Covid-19 Prediction Using Regression")
    st.sidebar.title("Choose the Options to get the output")
    st.markdown("Predicts the total number of cases after 10 days from the given date 😷")
    st.sidebar.markdown("Predicts the total number of cases after 10 days from the given date 😷")
    
    
    def load() :
         dataset = pd.read_csv('C:/Users/Abhijeet/Desktop/Semester 6/data mining/New folder/DATASETtest.csv')       
         return dataset
     
    def split(data,random_state) :
        y = data.CasesAfter10days
        x = data.drop(columns = ['CasesAfter10days'])
        x_train, x_test,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=random_state)
        return x_train, x_test,y_train , y_test

    def scale(x_train, x_test):
            sc_X = preprocessing.StandardScaler()
            scaled_features_x_train = sc_X.fit_transform(x_train.values)
            scaled_features_x_test = sc_X.transform(x_test.values)
            scaled_x_train = pd.DataFrame(scaled_features_x_train, index=x_train.index, columns=x_train.columns)
            scaled_x_test = pd.DataFrame(scaled_features_x_test, index=x_test.index, columns=x_test.columns)
            return scaled_x_train, scaled_x_test,sc_X        
    
    def plotingCurve(curve_list, x_train, x_test, y_train, y_test, y_pred):
        val = x_train['dayafter31dec']
        if 'Days vs total Confirmed' in curve_list:
            plt.scatter(x_train['dayafter31dec'],x_train['totalconfirmed'],color='magenta',)
            plt.title('Days vs total Confirmed')  
            plt.legend(labels=['total Confirmed'])
            plt.xlabel('Days')  
            plt.ylabel('totalConfirmed')  
            st.pyplot()
            
        if 'Days vs Training Data(Output set only)' in curve_list:
            plt.plot(val,y_train,'bs')
            plt.title('Days vs Training Data(Output set only)')
            plt.legend(labels=['Actual Train cases'])
            plt.xlabel('Days')  
            plt.ylabel('Cases after 10 days')
            st.pyplot()
        
        if 'Days vs ( Prediction and Actual cases(Testing data) )' in curve_list:
            
            val = x_test['dayafter31dec']
            pr = pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            #st.write('\t\t Comparision of predicted and actual data\n')
            st.markdown("<h3 style='text-align: center; color: black;'>Comparision of predicted and actual data</h3>", unsafe_allow_html=True)
            st.line_chart(pr)                
            
                
            plt.plot(val,y_test,'bs',val,y_pred,'g^')
            plt.title('Days vs ( Prediction and Actual cases(Testing data) )\n')
            plt.legend(labels=['Actual', 'Predict'])
            plt.xlabel('Days')  
            plt.ylabel('Cases after 10 days')
            plt.show()
            st.pyplot()
    
    def showPrediction(sc_X,scaling_option,model):
        dataset = pd.read_csv('C:/Users/Abhijeet/Desktop/Semester 6/data mining/New folder/Covid19_10_days_Input.csv')
        lastRow = dataset.tail(1)
        inputColumns=lastRow.drop(columns = ['CasesAfter10days'])
        referencedate = inputColumns['dayafter31dec']
        val=int(referencedate[referencedate.index.start])
        dateOfdata = datetime.date(2019,12,31)+datetime.timedelta(days=val)
        projectedDate=dateOfdata+datetime.timedelta(days=10)
        if(scaling_option=="Yes"):
            inputColumns_features=sc_X.transform(inputColumns.values)
            inputColumns = pd.DataFrame(inputColumns_features, index=inputColumns.index, columns=inputColumns.columns)
        
        ExpectedNoOfCases = model.predict(inputColumns)
        st.write("Expected Number of cases on "+str(projectedDate)+" predicted by given model is :"+str(ExpectedNoOfCases))
        
    def showPrediction2(sc_X,scaling_option,model,poly):
        dataset = pd.read_csv('C:/Users/Abhijeet/Desktop/Semester 6/data mining/New folder/Covid19_10_days_Input.csv')
        lastRow = dataset.tail(1)
        st.write(lastRow)
        inputColumns=lastRow.drop(columns = ['CasesAfter10days'])
        referencedate = inputColumns['dayafter31dec']
        val=int(referencedate[referencedate.index.start])
        dateOfdata = datetime.date(2019,12,31)+datetime.timedelta(days=val)
        projectedDate=dateOfdata+datetime.timedelta(days=10)
        
        if(scaling_option=="Yes"):
            inputColumns_features=sc_X.transform(inputColumns.values)
            inputColumns = pd.DataFrame(inputColumns_features, index=inputColumns.index, columns=inputColumns.columns)
        inputColumns = poly.transform(inputColumns)
        ExpectedNoOfCases = model.predict(inputColumns)
        st.write("Expected Number of cases on "+str(projectedDate)+" predicted by given model is :"+str(ExpectedNoOfCases))
        
        
    def regression(sc_X,scaling_option,regression_model,x_train, x_test,y_train , y_test,orig_x_train, orig_x_test):
        if(regression_model == 'MultipleLinearModel'):
            
            curves = st.sidebar.multiselect("What curve to plot ?",('Days vs total Confirmed','Days vs Training Data(Output set only)','Days vs ( Prediction and Actual cases(Testing data) )'))
            if st.sidebar.button("Regression",key="regression"):
                regr = linear_model.LinearRegression()
                regr.fit(x_train, y_train)
                y_pred = regr.predict(x_test)
                #print('Mean absolute error: %.2f'%mean_absolute_error(y_test,y_pred))
                showPrediction(sc_X,scaling_option,regr)
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                st.write('Mean absolute error: ',mean_absolute_error(y_test,y_pred).round(2))
                st.write("Root mean squared error : ",RMSE.round(2))
                plotingCurve(curves,orig_x_train, orig_x_test, y_train, y_test,y_pred)
        
        if(regression_model == 'Polynomial'):
            degree = st.sidebar.slider("What degree you want(a higher degree than 3 is taking too much time)?",2,4,key='degree')
            curves = st.sidebar.multiselect("What curve to plot ?",('Days vs total Confirmed','Days vs Training Data(Output set only)','Days vs ( Prediction and Actual cases(Testing data) )'))
            if st.sidebar.button("Regression",key="regression"):
                
                poly_degree = PolynomialFeatures(degree=degree,interaction_only=True)
                x_train = poly_degree.fit_transform(x_train)
                poly_degree.fit(x_train,y_train)
                
                x_test = poly_degree.fit_transform(x_test)
                
                
                regr = linear_model.LinearRegression()
                regr.fit(x_train,y_train)
                y_pred = regr.predict(x_test)
                showPrediction2(sc_X,scaling_option,regr,poly_degree)
                
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                st.write('Mean absolute error: ',mean_absolute_error(y_test,y_pred).round(2))
                st.write("Root mean squared error : ",RMSE.round(2))
                plotingCurve(curves,orig_x_train, orig_x_test, y_train, y_test,y_pred)                

        
        if(regression_model == 'SupportVectorRegression(SVR)'):
            max_itr=st.sidebar.number_input("Maximum number of iteration : ",100,10000,step=50,key='max_itr')
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01 , 10.0, step=0.01,key='C_LR')
            curves = st.sidebar.multiselect("What curve to plot ?",('Days vs total Confirmed','Days vs Training Data(Output set only)','Days vs ( Prediction and Actual cases(Testing data) )'))
            if st.sidebar.button("Regression",key="regression"):
                linearsvr = LinearSVR(random_state=0, tol=1e-5,max_iter=max_itr,C=C)
                linearsvr.fit(x_train,y_train)
                y_pred = linearsvr.predict(x_test)
                
                showPrediction(sc_X,scaling_option,linearsvr)
                
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                st.write('Mean absolute error: ',mean_absolute_error(y_test,y_pred).round(2))
                st.write("Root mean squared error : ",RMSE.round(2))
                plotingCurve(curves,orig_x_train, orig_x_test, y_train, y_test,y_pred) 
        
        if(regression_model == 'DecisionTree'):
            criterion = st.sidebar.radio("Criteria: for quality of a split ",("mse","friedman_mse", "mae"),key="criterion")
            max_features=st.sidebar.radio("max_features: number of features to consider for the best split ",("sqrt","auto","log2"),key="max_features")
            splitter = st.sidebar.radio("splitter: strategy used to choose the split at each node : ",("best","random"),key="splitter")
            curves = st.sidebar.multiselect("What curve to plot ?",('Days vs total Confirmed','Days vs Training Data(Output set only)','Days vs ( Prediction and Actual cases(Testing data) )'))
            if st.sidebar.button("Regression",key="regression"):
                Dt = DecisionTreeRegressor(max_features=max_features,splitter=splitter,criterion=criterion)
                Dt.fit(x_train,y_train)
                y_pred = Dt.predict(x_test)
                showPrediction(sc_X,scaling_option,Dt)
                
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                st.write('Mean absolute error: ',mean_absolute_error(y_test,y_pred).round(2))
                st.write("Root mean squared error : ",RMSE.round(2))
                plotingCurve(curves,orig_x_train, orig_x_test, y_train, y_test,y_pred) 
        
        if(regression_model == 'RandomForest'):
            n_estimators=st.sidebar.number_input("n_estimators: number of trees in forest",10,1000,step=10,key="n_estimators")
            criterion = st.sidebar.radio("Criteria: for quality of a split ",("mse", "mae"),key="criterion")
            max_features=st.sidebar.radio("max_features: number of features to consider for the best split ",("sqrt","auto","log2"),key="max_features")
            curves = st.sidebar.multiselect("What curve to plot ?",('Days vs total Confirmed','Days vs Training Data(Output set only)','Days vs ( Prediction and Actual cases(Testing data) )'))
            if st.sidebar.button("Regression",key="regression"):
                randomForest = RandomForestRegressor(n_estimators=n_estimators,criterion=criterion,max_features=max_features)
                randomForest.fit(x_train,y_train)
                y_pred = randomForest.predict(x_test)
                showPrediction(sc_X,scaling_option,randomForest)
                
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                st.write('Mean absolute error: ',mean_absolute_error(y_test,y_pred).round(2))
                st.write("Root mean squared error : ",RMSE.round(2))
                plotingCurve(curves,orig_x_train, orig_x_test, y_train, y_test,y_pred)
                
    data = load()
    random_state=st.sidebar.slider("Random state : ",min_value=1-1,max_value=43,step=1)
    x_train, x_test,y_train , y_test = split(data,random_state)
    orig_x_train, orig_x_test = x_train, x_test
    scaling_option = "Yes"
    scaling_option = st.sidebar.radio("Want to scale the data?",("Yes","No"),key="scaling_option")
    sc_X=0
    if(scaling_option == "Yes"):
        x_train,x_test,sc_X = scale(x_train, x_test)   
    st.sidebar.subheader("Choose Regressor")
    regression_model = st.sidebar.selectbox("Regressor",("MultipleLinearModel","Polynomial","SupportVectorRegression(SVR)","DecisionTree","RandomForest"))
    regression(sc_X,scaling_option,regression_model,x_train,x_test,y_train ,y_test,orig_x_train,orig_x_test)
    
    if st.sidebar.checkbox('show raw data',False):
        st.subheader("Covid-19(India) dataset")
        st.write(data)  
    
    
main()    