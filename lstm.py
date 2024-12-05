import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def prepare_data(df, column='Close'):
    data = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

def predict_next_n_days(model, last_sequence, scaler, n_days):
    next_n_days = []
    for _ in range(n_days):
        next_pred = model.predict(last_sequence.reshape(1, last_sequence.shape[0], 1))
        next_n_days.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    return scaler.inverse_transform(np.array(next_n_days).reshape(-1, 1))

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(df, y_true, y_pred, title, start_index):
    plt.figure(figsize=(16, 8))
    plt.plot(df.index[start_index:], y_true, label='Actual Price')
    plt.plot(df.index[start_index:], y_pred, label='Predicted Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_future_predictions(df, next_n_days, n_days):
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    plt.figure(figsize=(16, 8))
    plt.plot(df.index[-100:], df['Close'].values[-100:], label='Historical Price')
    plt.plot(future_dates, next_n_days, label='Predicted Price')
    plt.title(f'Stock Price Prediction (Next {n_days} Days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def stock_price_prediction(ticker, start_date, end_date, sequence_length, train_split, epochs, batch_size, validation_split, future_days):
    # Download stock data
    df = download_stock_data(ticker, start_date, end_date)

    # Prepare the data
    scaled_data, scaler = prepare_data(df)

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into training and testing sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    history = train_model(model, X_train, y_train, epochs, batch_size, validation_split)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MSE and MAE
    train_mse = mean_squared_error(y_train_inv, train_predictions)
    train_mae = mean_absolute_error(y_train_inv, train_predictions)
    test_mse = mean_squared_error(y_test_inv, test_predictions)
    test_mae = mean_absolute_error(y_test_inv, test_predictions)

    print(f"Train MSE: {train_mse:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")

    # Plot training history
    plot_training_history(history)

    # Plot predictions
    plot_predictions(df, y_train_inv, train_predictions, f'{ticker} Stock Price Prediction (Training Set)', sequence_length)
    plot_predictions(df, y_test_inv, test_predictions, f'{ticker} Stock Price Prediction (Test Set)', train_size + sequence_length)

    # Predict the next n days
    last_sequence = scaled_data[-sequence_length:]
    next_n_days = predict_next_n_days(model, last_sequence, scaler, future_days)

    # Plot future predictions
    plot_future_predictions(df, next_n_days, future_days)

    return model, scaler