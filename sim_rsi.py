import requests
import pandas as pd
import time
from agent_rsi import RSIAgent

# Function to fetch historical data
def fetch_historical_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.rename(columns={'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'Close', 'volume': 'VOLUME'}, inplace=True)
    return df

# Function to prepare the data
def prepare_data(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

# Function to split data into training and testing sets
def train_test_split(data, test_size=0.4):
    split_index = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    return train_data, test_data

# Backtesting function
def backtest(agent, data):
    #agent.train_model(data)
    portfolio_values = []  # List to store portfolio values over time
    
    for timestamp, row in data.iterrows():
        agent.trade(data.loc[:timestamp])
        portfolio_values.append(agent.get_portfolio_value(data.loc[timestamp, 'Close']))
    
    final_portfolio_value = agent.get_portfolio_value(data['Close'].iloc[-1])
    return portfolio_values, final_portfolio_value

# Function to calculate Sharpe Ratio, Total Return, and Max Drawdown
def calculate_metrics(portfolio_values):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    sharpe_ratio = returns.mean() / returns.std()
    cumulative_returns = pd.Series(portfolio_values) / portfolio_values[0]
    max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min() * 100
    return sharpe_ratio, total_return, max_drawdown

# Main function to run the backtesting
def run_backtesting(symbol, interval, agent_class):
    df = fetch_historical_data(symbol, interval)
    df = prepare_data(df)
    train_data, test_data = train_test_split(df)

    agent = agent_class()
    agent.train_model(train_data)
    train_portfolio_values, train_portfolio_value = backtest(agent, train_data)
    agent.cash=100000
    agent.holdings=0
    agent.position=0
    test_portfolio_values, test_portfolio_value = backtest(agent, test_data)

    train_metrics = calculate_metrics(train_portfolio_values)
    test_metrics = calculate_metrics(test_portfolio_values)

    return train_metrics, test_metrics, train_portfolio_value, test_portfolio_value

# Run backtesting for different intervals
intervals = ['1m', '1h', '4h', '1d']
agents = [RSIAgent, RSIAgent, RSIAgent, RSIAgent]

for interval, agent_class in zip(intervals, agents):
    train_metrics, test_metrics, train_portfolio_value, test_portfolio_value = run_backtesting('BTCUSDT', interval, agent_class)
    
    print(f"Metrics for {interval} Interval (Train):")
    print(f"Sharpe Ratio: {train_metrics[0]}")
    print(f"Total Return: {train_metrics[1]:.2f}%")
    print(f"Max Drawdown: {train_metrics[2]:.2f}%\n")
    
    print(f"Metrics for {interval} Interval (Test):")
    print(f"Sharpe Ratio: {test_metrics[0]}")
    print(f"Total Return: {test_metrics[1]:.2f}%")
    print(f"Max Drawdown: {test_metrics[2]:.2f}%\n")
    
    print(f"Portfolio Value for {interval} Interval (Train): {train_portfolio_value}")
    print(f"Portfolio Value for {interval} Interval (Test): {test_portfolio_value}")

# Function to update agents in real-time
def update_data(data, interval):
    new_data = fetch_historical_data('BTCUSDT', interval, limit=1)
    return pd.concat([data, new_data])

def update_agents():
    global df_1m, df_1h, df_4h, df_1d
    df_1m = update_data(df_1m, '1m')
    df_1h = update_data(df_1h, '1h')
    df_4h = update_data(df_4h, '4h')
    df_1d = update_data(df_1d, '1d')
    agent_1m.trade(df_1m)
    agent_1h.trade(df_1h)
    agent_4h.trade(df_4h)
    agent_1d.trade(df_1d)
    print(f"1m Interval Portfolio Value: {agent_1m.get_portfolio_value(df_1m['Close'].iloc[-1])}")
    print(f"1h Interval Portfolio Value: {agent_1h.get_portfolio_value(df_1h['Close'].iloc[-1])}")
    print(f"4h Interval Portfolio Value: {agent_4h.get_portfolio_value(df_4h['Close'].iloc[-1])}")
    print(f"1d Interval Portfolio Value: {agent_1d.get_portfolio_value(df_1d['Close'].iloc[-1])}")
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_4h = fetch_historical_data('BTCUSDT', '4h')
df_1d = fetch_historical_data('BTCUSDT', '1d')
df_1m = prepare_data(df_1m)
df_1h = prepare_data(df_1h)
df_4h = prepare_data(df_4h)
df_1d = prepare_data(df_1d)
agent_1m = RSIAgent()
agent_1h = RSIAgent()
agent_4h = RSIAgent()
agent_1d = RSIAgent()
while True:
    update_agents()
    time.sleep(5)  # Adjust the sleep time according to the interval