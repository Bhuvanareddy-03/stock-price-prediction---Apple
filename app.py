import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Apple Stock Price Forecasting App")
st.markdown("Compare ARIMA, SARIMA, XGBoost, and LSTM models to forecast Apple stock prices.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Apple Stock CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df = df.fillna(method='ffill')

    st.subheader("📊 Historical Close Price")
    st.line_chart(df['Close'])

    # Time Series Split
    train_size = int(len(df) * 0.9)
    train, test = df['Close'][:train_size], df['Close'][train_size:]

    # ARIMA
    arima_model = ARIMA(train, order=(5,1,0)).fit()
    arima_pred = arima_model.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    arima_pred.index = test.index
    mse_arima = mean_squared_error(test, arima_pred)
    mae_arima = mean_absolute_error(test, arima_pred)
    rmse_arima = sqrt(mse_arima)
    r2_arima = r2_score(test, arima_pred)

    # SARIMA
    sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    sarima_pred = sarima_model.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    sarima_pred.index = test.index
    mse_sarima = mean_squared_error(test, sarima_pred)
    mae_sarima = mean_absolute_error(test, sarima_pred)
    rmse_sarima = sqrt(mse_sarima)
    r2_sarima = r2_score(test, sarima_pred)

    # XGBoost
    df_ml = df.copy()
    df_ml['Day'] = df_ml.index.day
    df_ml['Month'] = df_ml.index.month
    df_ml['Year'] = df_ml.index.year
    df_ml['MA_5'] = df_ml['Close'].rolling(5).mean()
    df_ml['MA_10'] = df_ml['Close'].rolling(10).mean()
    df_ml = df_ml.dropna()
    X = df_ml[['Open','High','Low','Volume','Day','Month','Year','MA_5','MA_10']]
    y = df_ml['Close']
    split = int(len(X)*0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_test_index = X_test.index
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    mse_xgb = mean_squared_error(y_test, xgb_pred)
    mae_xgb = mean_absolute_error(y_test, xgb_pred)
    rmse_xgb = sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, xgb_pred)

    # LSTM
    data = df[['Close']]
    scaler_lstm = MinMaxScaler()
    scaled_data = scaler_lstm.fit_transform(data)
    train_size = int(len(scaled_data) * 0.9)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
    X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    lstm_pred = model.predict(X_test_lstm)
    lstm_pred = scaler_lstm.inverse_transform(lstm_pred)
    y_test_actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1,1))
    mse_lstm = mean_squared_error(y_test_actual, lstm_pred)
    mae_lstm = mean_absolute_error(y_test_actual, lstm_pred)
    rmse_lstm = sqrt(mse_lstm)
    r2_lstm = r2_score(y_test_actual, lstm_pred)

    # Model Comparison Table
    results_df = pd.DataFrame({
        'Model': ['ARIMA', 'SARIMA', 'XGBoost', 'LSTM'],
        'MSE': [mse_arima, mse_sarima, mse_xgb, mse_lstm],
        'MAE': [mae_arima, mae_sarima, mae_xgb, mae_lstm],
        'RMSE': [rmse_arima, rmse_sarima, rmse_xgb, rmse_lstm],
        'R²': [r2_arima, r2_sarima, r2_xgb, r2_lstm]
    }).round(4)

    st.subheader("📋 Model Performance Comparison")
    st.dataframe(results_df)

    # 🔘 Model Selection Dropdown
    st.subheader("🔘 Choose a Model to Visualize Forecast")
    model_choice = st.selectbox("Select a model", ['ARIMA', 'SARIMA', 'XGBoost', 'LSTM'])

    st.subheader(f"📈 {model_choice} Forecast Visualization")
    if model_choice == 'ARIMA':
        st.line_chart(pd.DataFrame({'Actual': test, 'ARIMA Forecast': arima_pred}))
    elif model_choice == 'SARIMA':
        st.line_chart(pd.DataFrame({'Actual': test, 'SARIMA Forecast': sarima_pred}))
    elif model_choice == 'XGBoost':
        xgb_pred_series = pd.Series(xgb_pred, index=X_test_index)
        y_test_series = pd.Series(y_test, index=X_test_index)
        st.line_chart(pd.DataFrame({'Actual': y_test_series, 'XGBoost Forecast': xgb_pred_series}))
    else:
        lstm_df = pd.DataFrame({'Actual': y_test_actual.flatten(), 'LSTM Forecast': lstm_pred.flatten()})
        st.line_chart(lstm_df)

    # 📅 30-Day Forecast
    st.subheader("📅 30-Day Future Forecast")
    if model_choice == 'SARIMA':
        future_forecast = sarima_model.forecast(steps=30)
    elif model_choice == 'ARIMA':
        future_forecast = arima_model.forecast(steps=30)
    elif model_choice == 'XGBoost':
        future_forecast = xgb_model.predict(X_test_scaled[-30:])
    else:
        last_60 = scaled_data[-60:]
        temp_input = list(last_60.reshape(-1))
        lst_output = []
        for i in range(30):
            X_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
            yhat = model.predict(X_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
        future_forecast = scaler_lstm.inverse_transform(np.array(lst_output).reshape(-1,1)).flatten()

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_forecast}).set_index('Date
