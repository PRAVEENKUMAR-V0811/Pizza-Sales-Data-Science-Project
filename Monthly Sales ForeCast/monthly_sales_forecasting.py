import pandas as pd
from prophet import Prophet

def load_and_prepare_data(filepath):
    # Load CSV without parsing dates
    df = pd.read_csv(filepath)

    # Convert 'order_date' column explicitly to datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Drop rows where 'order_date' conversion failed (if any)
    df = df.dropna(subset=['order_date'])

    # Now you can use dt accessor safely
    df['YearMonth'] = df['order_date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('YearMonth').agg({'quantity': 'sum'}).reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales['YearMonth'])
    monthly_sales = monthly_sales.rename(columns={'Date': 'ds', 'quantity': 'y'})

    return monthly_sales[['ds', 'y']]

def train_and_forecast(monthly_sales, periods=12):
    # Initialize Prophet model
    model = Prophet()

    # Fit the model on historical data
    model.fit(monthly_sales)

    # Create future dataframe for next 'periods' months
    future = model.make_future_dataframe(periods=periods, freq='ME')

    # Predict future sales
    forecast = model.predict(future)

    return forecast

def save_forecast(forecast, filename='monthly_sales_forecast_2016.csv'):
    # Extract only the forecast for the future periods (last 12 months)
    forecast_to_save = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).copy()

    # Rename columns for clarity in output CSV
    forecast_to_save = forecast_to_save.rename(columns={
        'ds': 'Forecast_Month',
        'yhat': 'Predicted_Sales_Quantity',
        'yhat_lower': 'Lower_Bound',
        'yhat_upper': 'Upper_Bound'
    })

    # Save to CSV
    forecast_to_save.to_csv(filename, index=False)
    print(f"Forecast saved to {filename}")

def main():
    data_path = '../pizza_sales.csv'

    monthly_sales = load_and_prepare_data(data_path)
    forecast = train_and_forecast(monthly_sales, periods=12)
    save_forecast(forecast)

if __name__ == "__main__":
    main()
