import requests
import pandas as pd
import time
from agent_ma import MovingAverageCrossoverAgent
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
# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_4h = fetch_historical_data('BTCUSDT', '4h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# Prepare the training data
def prepare_data(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data
df_1m = prepare_data(df_1m)
df_1h = prepare_data(df_1h)
df_4h = prepare_data(df_4h)
df_1d = prepare_data(df_1d)

# Backtesting function
def backtest(agent, data):
    agent.train_model(data)
    portfolio_values = []  # List to store portfolio values over time
    
    for timestamp, row in data.iterrows():
        agent.trade(data.loc[:timestamp])
        portfolio_values.append(agent.get_portfolio_value(data.loc[timestamp, 'Close']))
    
    final_portfolio_value = agent.get_portfolio_value(data['Close'].iloc[-1])
    return portfolio_values, final_portfolio_value

# Initialize agents
agent_1m = MovingAverageCrossoverAgent()
agent_1h = MovingAverageCrossoverAgent()
agent_4h = MovingAverageCrossoverAgent()
agent_1d = MovingAverageCrossoverAgent()
# Backtest each agent
# Backtest each Moving Average Crossover agent and get portfolio values
portfolio_values_1m, portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_values_1h, portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_values_4h, portfolio_value_4h = backtest(agent_4h, df_4h)
portfolio_values_1d, portfolio_value_1d = backtest(agent_1d, df_1d)
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 4h Interval: {portfolio_value_4h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

# Function to calculate Sharpe Ratio, Total Return, and Max Drawdown
def calculate_metrics(portfolio_values):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    sharpe_ratio = returns.mean() / returns.std()
    cumulative_returns = pd.Series(portfolio_values) / portfolio_values[0]
    max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min() * 100
    return sharpe_ratio, total_return, max_drawdown

# Calculate metrics for each interval
sharpe_1m, total_return_1m, max_drawdown_1m = calculate_metrics(portfolio_values_1m)
sharpe_1h, total_return_1h, max_drawdown_1h = calculate_metrics(portfolio_values_1h)
sharpe_4h, total_return_4h, max_drawdown_4h = calculate_metrics(portfolio_values_4h)
sharpe_1d, total_return_1d, max_drawdown_1d = calculate_metrics(portfolio_values_1d)

# Print results
print(f"Metrics for 1m Interval:")
print(f"Sharpe Ratio: {sharpe_1m}")
print(f"Total Return: {total_return_1m:.2f}%")
print(f"Max Drawdown: {max_drawdown_1m:.2f}%\n")

print(f"Metrics for 1h Interval:")
print(f"Sharpe Ratio: {sharpe_1h}")
print(f"Total Return: {total_return_1h:.2f}%")
print(f"Max Drawdown: {max_drawdown_1h:.2f}%\n")

print(f"Metrics for 4h Interval:")
print(f"Sharpe Ratio: {sharpe_4h}")
print(f"Total Return: {total_return_4h:.2f}%")
print(f"Max Drawdown: {max_drawdown_4h:.2f}%\n")

print(f"Metrics for 1d Interval:")
print(f"Sharpe Ratio: {sharpe_1d}")
print(f"Total Return: {total_return_1d:.2f}%")
print(f"Max Drawdown: {max_drawdown_1d:.2f}%")

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
while True:
    update_agents()
    time.sleep(5) # Adjust the sleep time according to the interval