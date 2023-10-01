from linear_regression import LinearRegression
import streamlit as st
from helper import *
import pickle
from datetime import date, timedelta

st.title('Predicting number of receipts using Linear Regression')
df = handlingInput('data_daily.csv')

option  = st.selectbox("Select One",["View Data","Plot Data","Training Model","Inference","Visualize Results"])
if option == "View Data":
    st.subheader("View Data")
    st.markdown('We have added the `month`, `day of the month`,`weekend` and `time step` columns to help better understand the data')
    sideselectbox = st.sidebar.selectbox("View One",["Head","Tail","Resample"])
    if sideselectbox == "Head":
        h = st.sidebar.number_input('Size of Head', 1, 365, 5)
        st.write(df.head(h),use_container_width=True)           # type: ignore
    elif sideselectbox == "Tail":
        t = st.sidebar.number_input('Size of Tail', 1, 365, 5)
        st.write(df.tail(t),use_container_width=True)       # type: ignore
    else:
        r = st.sidebar.number_input('Size of Tail', 1, 365, 50)
        st.write(df.sample(n=r),use_container_width=True) # type: ignore

elif option == "Plot Data":
    sideselectbox = st.sidebar.multiselect("Plot One",["Entire Data","Monthly Data","Weekend Data"], "Entire Data")
    st.subheader('Plot Data to see any visual Trends')
    if "Entire Data" in sideselectbox:
        st.plotly_chart(plotData(df), use_container_width=True)
        st.markdown('From the plot we can see a linear relationship between Date and Receipts')
    
    if "Monthly Data" in sideselectbox:
        st.plotly_chart(plotMonthlyData(df), use_container_width=True)
        st.markdown('Although there seems to be some relation between the same days over the month but it is not significant.')
    
    if "Weekend Data" in sideselectbox:
        st.plotly_chart(plotDataWeekends(df),use_container_width=True)
        st.markdown('We do not visual see any trend that more people are uploading receipts on the weekends compared to week days.')


elif option == "Training Model":
    st.subheader('Model Finalization')
    st.markdown('Observed things from Data:-')
    st.markdown('1. Linear Relationship')
    st.markdown('2. No Significant Monthly or Weekend Trends')
    st.markdown('3. From the plot we can see that data shows Homoscedasticity')
    st.markdown('Based on the above observations and data following assumptions the best model we can select is Linear Regression')

    X_train, X_test, y_train, y_test, X, y = preProcessing(df)
    regressor = LinearRegression(learning_rate=0.000035, epochs=1000000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    st.markdown(f'After training, we get weight and bias as follows {regressor.W[0]}`,`{round(regressor.b)}`') # type: ignore

elif option == "Inference":
    regressor = pickle.load(open('model.pkl', 'rb'))
    start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2025, 12, 31))

    # Calculate the number of days elapsed since 1st Jan 2021 for the start date
    days_elapsed = (start_date - date(2021, 1, 1)).days # type: ignore

    # Generate date range between start and end dates
    date_range = pd.date_range(start=start_date, end=end_date, freq="D") # type: ignore

    # Create a DataFrame from the date range with an index column
    date_df = pd.DataFrame({"Date": date_range, "Day": range(days_elapsed, days_elapsed + len(date_range))})
    sideselectbox = st.sidebar.selectbox("Inference?",["Daily","Monthly"])
    prediction = regressor.predict(np.array(date_df['Day']).reshape(-1,1))
    date_df['Prediction'] = list(map(round,prediction))

    if sideselectbox == "Daily":
        data_as_csv= date_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download data as CSV",data=data_as_csv,file_name='Predictions.csv',mime='text/csv') # type: ignore
    else:
        monthly_sum = date_df.resample(rule='M', on='Date')['Prediction'].sum()
        data_as_csv= monthly_sum.to_csv(index=False).encode("utf-8")       
        st.download_button(label="Download data as CSV",data=data_as_csv,file_name='Predictions.csv',mime='text/csv')
else:
    regressor = pickle.load(open('model.pkl', 'rb'))
    plotCalculations(regressor,df)
    X_train, X_test, y_train, y_test, X, y = preProcessing(df)
    st.plotly_chart(plotRegressionLine(X_train, X_test, y_train, y_test, X, regressor))

    st.markdown('We can then extrapolate the regression line to calculate the sum of each month for the year `2022` represented by green marker')

    st.plotly_chart(plotMonthlySum(regressor, df)[0], theme="streamlit", use_container_width=True)
    st.subheader('Prediction: Total Number of scanned receipts for each month in 2022')
    st.write(plotMonthlySum(regressor, df)[2])