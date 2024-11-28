from your_dashboard.computation_data_app.cache_manager import fetch_get_data  # Assuming you have this module
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import asyncio

# Initialize Binance client
api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

# Asynchronous function to fetch historical klines and save to CSV
async def fetch_and_save_historical_klines(symbol, interval, filename):
    end_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    start_str = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
    
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    # Extract necessary columns from klines data
    klines_data = [[row[0], row[6], row[4]] for row in klines]  # open_time, close_time, close

    # Create DataFrame from the extracted data
    df_klines = pd.DataFrame(klines_data, columns=['open_time', 'close_time', 'close'])

    # Convert timestamp to datetime
    df_klines['open_time'] = pd.to_datetime(df_klines['open_time'], unit='ms')
    df_klines['close_time'] = pd.to_datetime(df_klines['close_time'], unit='ms')
    df_klines['close'] = df_klines['close'].astype(float)

    # Save DataFrame to CSV
    df_klines.to_csv(filename, index=False)
# Function to calculate total fees using the saved CSV
def calculate_total_fees_from_csv(start_str, end_str, filename):
    # Load historical klines data from CSV
    df_klines = pd.read_csv(filename)
    df_klines['open_time'] = pd.to_datetime(df_klines['open_time']).dt.date  # Use only date for comparison

    # Load cached data (assuming fetch_get_data function is defined)
    cached_data = asyncio.run(fetch_get_data())
    df = pd.DataFrame(cached_data)

    # Convert 'date' column to datetime if not already done
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date  # Convert to date only

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

    # Initialize price_data dictionary to store prices by date
    price_data = {}
    for _, row in df_klines.iterrows():
        open_time = row['open_time']
        close_price = row['close']
        price_data[open_time] = close_price

    # Filter the DataFrame based on the specified date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_filtered = df.loc[mask]

    # Filter the relevant columns
    filtered_df = df_filtered[['date', 'commission', 'commissionAsset']]

    # Calculate total fees in USDT
    total_fees_usdt = 0.0
    for index, row in filtered_df.iterrows():
        date_only = row['date']  # Get the date portion
        if row['commissionAsset'] == 'BNB':
            price = price_data.get(date_only)
            if price is None:
                # Handle missing price data if necessary
                continue
            commission_usdt = row['commission'] * price
        elif row['commissionAsset'] == 'USDT':
            commission_usdt = row['commission']
        else:
            commission_usdt = 0

        total_fees_usdt += commission_usdt

    return total_fees_usdt, filtered_df.head()

# Run the asynchronous function to fetch and save historical klines
symbol = 'BNBUSDT'
interval = Client.KLINE_INTERVAL_4HOUR
filename = 'historical_klines.csv'
asyncio.run(fetch_and_save_historical_klines(symbol, interval, filename))

# Calculate total fees from the saved CSV
start_date = '2024-06-10'
end_date = '2024-06-15'
total_fees_usdt, filtered_df_head = calculate_total_fees_from_csv(start_date, end_date, filename)

# Display the results
print(f"Total Fees in USDT: {total_fees_usdt}")
print(filtered_df_head)