import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
from plotly import graph_objs as go


def handlingInput(csv_name):
    df = pd.read_csv(csv_name)
    df.columns = ['date','receipts']
    df.date = pd.to_datetime(df.date, format = '%Y-%m-%d')
    df['Month'] = pd.DatetimeIndex(df.date).month
    df['Day'] = pd.DatetimeIndex(df.date).day
    df['Time'] = np.arange(1,len(df.index)+1)
    df["Weekend"] = df.date.dt.day_name().isin(['Saturday', 'Sunday'])
    return df

def plotData(df):
    fig = px.line(df,x='date',y='receipts',markers=True)
    return fig

def plotMonthlyData(df):
    fig = px.line(df,x='Day',y='receipts',color='Month')
    return fig

def plotDataWeekends(df):
    fig = px.line(df,x='date',y='receipts',color='Weekend',markers=True)
    return fig

def preProcessing(df):
    X = np.array(df.Time).reshape(-1,1)
    y = np.array(df.receipts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
    return X_train, X_test, y_train, y_test, X, y

def plotRegressionLine(X_train, X_test, y_train, y_test, X, regressor):
    y_pred_line = regressor.predict(X)
    train = pd.DataFrame(X_train,y_train).reset_index()
    test = pd.DataFrame(X_test,y_test).reset_index()
    pred = pd.DataFrame(X,y_pred_line).reset_index()
    train.columns = ['receipts','time']
    test.columns = ['receipts', 'time']
    pred.columns = ['receipts', 'time']
    fig1 = px.scatter(train, x='time', y='receipts')
    fig2 = px.scatter(test, x='time', y='receipts').update_traces(marker=dict(color='orange'))
    fig3 = px.line(pred, x='time',y='receipts').update_traces(line_color='red', line_width=5)
    fig = go.Figure(data = fig1.data + fig2.data + fig3.data).update_layout( # type: ignore
        xaxis_title="Time Step", yaxis_title="No of receipts")
    return fig

def plotCalculations(regressor,df):
    y_pred_line = regressor.predict(np.array(list(range(0,730))).reshape(-1,1))
    pred = pd.DataFrame(y_pred_line[365:730],pd.date_range(start='01/01/2022', end='31/12/2022'))
    pred = pred.reset_index()
    pred.columns = ['date', 'receipts']
    
    monthly_sum = pred.resample(rule='M', on='date')['receipts'].sum()
    monthly_sum21 = df.resample(rule='M', on='date')['receipts'].sum()
    df21 = pd.DataFrame(pd.date_range(start='1/1/2021', periods=12, freq='M'), monthly_sum21).reset_index()
    df22 = pd.DataFrame(pd.date_range(start='1/1/2022', periods=12, freq='M'), monthly_sum).reset_index()
    df21.columns = ['receipts','time']
    df22.columns = ['receipts','time']
    df22['month'] = pd.DatetimeIndex(df22.time).month # type: ignore
    df22 = df22[['month','receipts','time']]
    return monthly_sum,monthly_sum21,y_pred_line,df21,df22

def plotMonthlySum(regressor, df):
    _,_,_,df21,df22 = plotCalculations(regressor,df)
    fig1 = px.scatter(df21, x='time', y='receipts')
    fig2 = px.scatter(df22, x='time', y='receipts').update_traces(marker=dict(color='red'))

    fig = go.Figure(data = fig1.data + fig2.data) # type: ignore
    del df22['time']
    df22 = df22.style.background_gradient(axis=0)
    
    return fig, df21, df22



