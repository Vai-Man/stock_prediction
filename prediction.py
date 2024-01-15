import streamlit as st
import numpy as np
import yfinance as yf

from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction')
stocks = st.text_input("Enter stock name")
n_days = st.number_input('Number of Days of Prediction:', min_value=1, value=30)

st.info("Kindly note that the application will present the stock value in the local currency of the selected stock.")

def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        
        if data.empty:
            st.error("Error: No data found for the specified stock.")
            return None

        data.reset_index(inplace=True)
        data['Date'] = np.array(data['Date'])

        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

data = load_data(stocks)

if data is not None:
    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_days, freq='D')
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(title_text=f'Forecast plot for {n_days} days', xaxis_rangeslider_visible=True)

    fig1.update_layout(
        xaxis_title='Date',
        yaxis_title='Close Price'
    )
    st.plotly_chart(fig1)

    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
