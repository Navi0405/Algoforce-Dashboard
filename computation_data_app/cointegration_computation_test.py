import pandas as pd
import os
from binance.client import Client
from datetime import datetime, timezone

#account2

starting_date = '2024-10-01 00:00:00'
ending_date = '2024-12-01 00:00:00'

# Binance API keys from environment variables
account2_key = os.environ.get('account2_key')
account2_secret = os.environ.get('account2_secret')

client_account2 = Client(account2_key, account2_secret)

# CSV file path
CSV_FILE_PATH = "..path/\\to/\\your/\\your/\\your_account\\tradesheet.csv"

# Time range for filtering data
STARTING_DATE = '2024-10-01 00:00:00'
ENDING_DATE = '2024-11-01 00:00:05'

account2_init_balance = 12215.2

account2_binance_data = "..path/\\to/\\your/\\your/\\your/\\file/\\your_dashboard\\accountfolder\\account2_4H_continuing_trades.csv"
your_account_binance_data = "..path/\\to/\\your/\\your/\\your/\\file/\\your_dashboard\\accountfolder\\accountyour_account_4H_continuing_trades.csv"
# your_account_binance_data = "..path/\\to/\\your/\\your/\\your/\\file/\\your_dashboard\\accountfolder\\your_account_4H_continuing_trades.csv"

def account2_read_csv_data():
    """Read CSV file and filter data between starting and ending dates."""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df["realizedPnl"] = (df.qty_0 * (df.exit_price_0 - df.entry_price_0)) + \
                            (df.qty_1 * (df.exit_price_1 - df.entry_price_1))
        filtered_df = df[(df.exit_dt >= STARTING_DATE) & (df.exit_dt <= ENDING_DATE)]
        return filtered_df
    except FileNotFoundError:
        raise Exception("CSV file not found.")

def account2_wallet_balance():
    """Fetch wallet balance and unrealized PNL from Binance account."""
    wallet_account = client_account2.futures_account()
    total_balance = float(wallet_account['totalWalletBalance'])
    unrealized_pnl = float(wallet_account['totalUnrealizedProfit'])
    total_balance += unrealized_pnl
    return total_balance, unrealized_pnl

def account2_count_position():
    """Count positions from Binance account."""
    position = pd.DataFrame(client_account2.futures_position_information())
    notional_size = position['positionAmt'].astype(float)
    long_count = (notional_size > 0).sum()
    short_count = (notional_size < 0).sum()
    return long_count + short_count, long_count, short_count

def account2_compute_trading_days_and_fees():
    """Compute trading days, winning days, total fees, and max drawdown from the CSV file."""
    
    try:
        # Read the CSV file
        tradelogs = pd.read_csv(account2_binance_data)

        # Ensure the 'date' column is in datetime format
        tradelogs['date'] = pd.to_datetime(tradelogs['date'])

        # Compute unique trading days by extracting unique dates from 'date' column
        trading_days = tradelogs['date'].dt.date.nunique()

        # Compute winning days based on 'realizedPnl' (assuming 'realizedPnl' > 0 means a win)
        winning_days = tradelogs[tradelogs['realizedPnl'] > 0]['date'].dt.date.nunique()

        # Compute total fees (using 'commission' column)
        total_fees = tradelogs['commission'].sum()

        # Calculate cumulative balance from the 'realizedPnl' column
        tradelogs['cumulative_balance'] = tradelogs['realizedPnl'].cumsum()

        # Compute peak balance at each time step
        tradelogs['peak_balance'] = tradelogs['cumulative_balance'].cummax()

        # Calculate drawdown as percentage drop from the peak balance
        tradelogs['drawdown'] = (tradelogs['cumulative_balance'] - tradelogs['peak_balance']) / tradelogs['peak_balance']

        # Max drawdown is the minimum of the drawdown column (most negative drawdown)
        max_drawdown = tradelogs['drawdown'].min()

        return {
            'Trading_Days': int(trading_days),
            'Winning_Days': int(winning_days),
            'Total_Fees': round(float(total_fees), 2),
            'Max_Drawdown': round(max_drawdown * 100, 2)  # Convert to percentage
        }

    except FileNotFoundError:
        return {"error": "CSV file not found. Please check the file path."}
    except KeyError as e:
        return {"error": f"Missing expected column in the CSV: {e}"}
    except Exception as e:
        return {"error": str(e)}

def account2_compute_metrics():
    """Compute various trading metrics from the CSV file and Binance account."""
    try:
        # Initial balance for calculations
        account_init_balance = 12257.04

        # Read and process the CSV file
        final_df = account2_read_csv_data()

        # Binance account balances
        total_balance, unrealized_pnl = account2_wallet_balance()

        # Position counts
        total_positions, long_count, short_count = account2_count_position()

        # Calculate realized PnL and other statistics
        num_trades = len(final_df)
        realized_pnl = final_df['realizedPnl'].sum()
        profit = total_balance - account_init_balance
        winrate = final_df[final_df['realizedPnl'] > 0].shape[0] / num_trades if num_trades > 0 else 0

        # Format and round the computed data to ensure compatibility with JSON
        computed_data = {
            'Account': 'account2',
            'total_trades': int(num_trades),
            'realized_pnl': round(float(realized_pnl), 2),
            'profit': round(float(profit), 2),
            'total_balance': round(float(total_balance), 2),
            'unrealized_pnl': round(float(unrealized_pnl), 2),
            'winrate': round(winrate * 100, 2),  # Convert to percentage
            'total_positions': int(total_positions),
            'long_positions': int(long_count),
            'short_positions': int(short_count)
        }

        return computed_data
    
    except Exception as e:
        return {"error": str(e)}


#accountyour_account

import mysql.connector

accountyour_account_key = os.environ.get('accountyour_account_key')
accountyour_account_secret = os.environ.get('accountyour_account_secret')

client_accountyour_account = Client(accountyour_account_key, accountyour_account_secret)

account_name = "accountyour_account"

your_account_init_balance = 103872

def your_account_wallet_balance():
    B_account, acc_name = client_accountyour_account, account_name
    wallet_account = B_account.futures_account()
    total_balance = float(wallet_account['totalWalletBalance'])
    unrealizedPnl = float(wallet_account['totalUnrealizedProfit'])
    total_balance = total_balance + unrealizedPnl

    return float(total_balance), float(unrealizedPnl)

def create_connection(database_name):
    connection = mysql.connector.connect(
        host="your_name-rds.cl6akmuiy6oy.eu-north-1.rds.amazonaws.com",
        your/="admin",
        password="your_account.1",
        database=database_name
    )

    return connection

def your_account_create_tradelogs():
    table_name = "your_account_TradeLogs"
    database_name = 'your_namedb'
    connection = create_connection(database_name)
    query = f"SELECT * FROM {table_name}"
    tradelogs = pd.read_sql(query, connection)
    connection.close()
    tradelogs.to_csv("..path/\\to/\\your/\\your/\\your/\\file/\\your_dashboard\\computation_data_app\\your_account_tradelogs.csv")

    return tradelogs

def your_account_compute_trading_days_and_fees():
    """Compute trading days, winning days, total fees, and max drawdown from the CSV file."""
    
    try:
        # Read the CSV file
        tradelogs = pd.read_csv(your_account_binance_data)

        # Ensure the 'date' column is in datetime format
        tradelogs['date'] = pd.to_datetime(tradelogs['date'])

        # Compute unique trading days by extracting unique dates from 'date' column
        trading_days = tradelogs['date'].dt.date.nunique()

        # Compute winning days based on 'realizedPnl' (assuming 'realizedPnl' > 0 means a win)
        winning_days = tradelogs[tradelogs['realizedPnl'] > 0]['date'].dt.date.nunique()

        # Compute total fees (using 'commission' column)
        total_fees = tradelogs['commission'].sum()

        # Calculate cumulative balance from the 'realizedPnl' column
        tradelogs['cumulative_balance'] = tradelogs['realizedPnl'].cumsum()

        # Compute peak balance at each time step
        tradelogs['peak_balance'] = tradelogs['cumulative_balance'].cummax()

        # Calculate drawdown as percentage drop from the peak balance
        tradelogs['drawdown'] = (tradelogs['cumulative_balance'] - tradelogs['peak_balance']) / tradelogs['peak_balance']

        # Max drawdown is the minimum of the drawdown column (most negative drawdown)
        max_drawdown = tradelogs['drawdown'].min()

        return {
            'Trading_Days': int(trading_days),
            'Winning_Days': int(winning_days),
            'Total_Fees': round(float(total_fees), 2),
            'Max_Drawdown': round(max_drawdown * 100, 2)  # Convert to percentage
        }

    except FileNotFoundError:
        return {"error": "CSV file not found. Please check the file path."}
    except KeyError as e:
        return {"error": f"Missing expected column in the CSV: {e}"}
    except Exception as e:
        return {"error": str(e)}

def your_account_compute_metrics():

    account_name = "accountyour_account"
    datenow = datetime.utcnow()

    # Get total balance and unrealized PnL for the account
    total_balance, unrealizedPnl = your_account_wallet_balance()

    tradelogs['wins'] = tradelogs['netting'].apply(lambda x: 1 if x>0 else 0)
    tradelogs = tradelogs[tradelogs['date'] >= starting_date]
    total_trades = len(tradelogs)
    p1_df = tradelogs.copy()
    p1_df = p1_df[p1_df['date'] >= starting_date]
    total_perc_return = round(((total_balance-your_account_init_balance)/your_account_init_balance) * 100, 2)
    total_profit_usd = round(your_account_init_balance * (total_perc_return/100),2)

    overall_winrate = round((tradelogs['wins'].sum() / len(tradelogs['wins'])), 2)
    print(overall_winrate, total_profit_usd, total_perc_return, total_trades)

    # Create a message dictionary with the results
    from dateutil import tz
    time_now = datetime.now(tz.gettz()).strftime("%m/%d/%Y, %H:%M")

    computed_data = {
        'Account': account_name,
        'total_trades': total_trades,
        'return': round(total_perc_return, 2),
        'profit': total_profit_usd,
        'winrate': round((overall_winrate) * 100, 2),
        'total_balance': round(total_balance, 2),
        'unrealized_pnl': round(unrealizedPnl, 2)
    }

    return computed_data

#your_account

import mysql.connector

your_account_key = os.environ.get('your_account_key')
your_account_secret = os.environ.get('your_account_secret')

client_your_account = Client(your_account_key, your_account_secret)

account_name = "your_account"

your_account_init_balance = 146697

def your_account_wallet_balance():
    B_account, acc_name = client_your_account, account_name
    wallet_account = B_account.futures_account()
    total_balance = float(wallet_account['totalWalletBalance'])
    unrealizedPnl = float(wallet_account['totalUnrealizedProfit'])
    total_balance = total_balance + unrealizedPnl

    return float(total_balance), float(unrealizedPnl)

def create_connection(database_name):
    connection = mysql.connector.connect(
        host="your_name-rds.cl6akmuiy6oy.eu-north-1.rds.amazonaws.com",
        your/="admin",
        password="your_account.1",
        database=database_name
    )

    return connection

def your_account_create_tradelogs():
    table_name = "TradeLogs"
    database_name = 'your_namedb1'
    connection = create_connection(database_name)
    query = f"SELECT * FROM {table_name}"
    tradelogs = pd.read_sql(query, connection)
    connection.close()
    tradelogs.to_csv("..path/\\to/\\your/\\your/\\your/\\file/\\your_dashboard\\computation_data_app\\tradelogs_your_account.csv")

    return tradelogs

def your_account_compute_metrics():

    account_name = "your_account"
    datenow = datetime.utcnow()

    # Get total balance and unrealized PnL for the account
    total_balance, unrealizedPnl = your_account_wallet_balance()

    # Create tradelogs (this will generate the CSV and return the DataFrame)
    tradelogs = your_account_create_tradelogs()

    if len(tradelogs) != 0:
        tradelogs['wins'] = tradelogs['netting'].apply(lambda x: 1 if x>0 else 0)
        tradelogs = tradelogs[tradelogs['exit_dt'] >= starting_date]
        total_trades = len(tradelogs)
        p1_df = tradelogs.copy()
        p1_df = p1_df[p1_df['exit_dt'] >= starting_date]
        perc_return = round(((total_balance-your_account_init_balance)/your_account_init_balance) * 100, 2)
        total_profit_usd = round(your_account_init_balance * (perc_return/100),2)
        
        overall_winrate = round((tradelogs['wins'].sum() / len(tradelogs['wins'])), 2)
        print(overall_winrate, total_profit_usd, perc_return, perc_return, total_trades)

        from dateutil import tz
        time_now = datetime.now(tz.gettz()).strftime("%m/%d/%Y, %H:%M")
        message_dict = {
                'Account': account_name,
                'total_trades': total_trades,
                'return': round(perc_return,2),
                'profit' : total_profit_usd,
                'winrate': round((overall_winrate)*100,2),
                'total_balance': round(total_balance,2),
                'unrealized_pnl': round(unrealizedPnl,2)
            }
        
        return message_dict
    
    else:
        # time_now = datetime.now(tz.gettz()).strftime("%m/%d/%Y, %H:%M")
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        time_now = now_utc.date()
        message_dict = {
        'Account': account_name,
        'total_trades': total_trades,
        'return': f"{round(0,2)}%",
        'profit' : f"$ {0.0}",
        'winrate': f"{round((0)*100,2)}%",
        'total_balance': f"$ {round(total_balance,2)}",
        'unrealized_pnl': f"$ {round(unrealizedPnl,2)}"
            
        }

        return message_dict



def compute_combined_metrics():
    """Compute combined metrics for account2, accountyour_account, and your_account accounts."""

    # Compute metrics for account2
    account2_data = account2_compute_metrics()
    account2_tradelogs_data = account2_compute_trading_days_and_fees()

    # Compute metrics for accountyour_account
    your_account_data = your_account_compute_metrics()
    your_account_tradelogs_data = your_account_compute_trading_days_and_fees()

    # Compute metrics for your_account
    your_account_data = your_account_compute_metrics()

    # Check for errors in CSV processing for account2 and accountyour_account
    if 'error' in account2_tradelogs_data:
        print(f"account2 CSV error: {account2_tradelogs_data['error']}")
        account2_tradelogs_data = {'total_trades': 0, 'Winning Days': 0, 'Total Fees': 0}

    if 'error' in your_account_tradelogs_data:
        print(f"accountyour_account CSV error: {your_account_tradelogs_data['error']}")
        your_account_tradelogs_data = {'total_trades': 0, 'Winning Days': 0, 'Total Fees': 0}

    # Combine the metrics
    combined_data = {
        'Account': 'ALL',
        'Total_Balance': account2_data['total_balance'] + your_account_data['total_balance'] + your_account_data['total_balance'],
        'Total_Realized_Pnl': account2_data['profit'] + your_account_data['profit'] + your_account_data['profit'],
        
        # Combined return calculation based on initial balances of each account
        'Return': (
            (account2_data['profit'] + your_account_data['profit'] + your_account_data['profit']) /
            (12257.04 + 105144.05 + 150000.0) * 100
        ),
        
        'Unrealized_PnL': account2_data['unrealized_pnl'] + your_account_data['unrealized_pnl'] + your_account_data['unrealized_pnl'],
        'Total_Trades': account2_data['total_trades'] + your_account_data['total_trades'] + your_account_data['total_trades'],

        # Weighted winrate calculation
        'Win_Rate': (
            (account2_data['winrate'] * account2_data['total_trades']) + 
            (your_account_data['winrate'] * your_account_data['total_trades']) +
            (your_account_data['winrate'] * your_account_data['total_trades'])
         / (
            account2_data['total_trades'] + your_account_data['total_trades'] + your_account_data['total_trades'])
        ),

        # Trading days and fees for account2 and accountyour_account
        'Trading_Days': account2_tradelogs_data['Trading_Days'] + your_account_tradelogs_data['Trading_Days'],
        'Winning_Days': account2_tradelogs_data['Winning_Days'] + your_account_tradelogs_data['Winning_Days'],
        'Total_Fees': account2_tradelogs_data['Total_Fees'] + your_account_tradelogs_data['Total_Fees'],
    }

    # Display individual results for debugging
    print("account2 Metrics:", account2_data)
    print("account Metrics:", your_account_data)
    print("account Metrics:", your_account_data)
    print("Combined Data:", combined_data)

    return combined_data
