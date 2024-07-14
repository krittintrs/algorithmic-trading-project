from agent_super import TradingAgent
class MovingAverageCrossoverAgent(TradingAgent):
    def __init__(self, short_window=50, long_window=200, initial_cash=100000):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window
    def generate_signals(self, data):
        data = data.copy()  # Avoid SettingWithCopyWarning
        data.loc[:, 'SMA50'] = data['Close'].rolling(window=self.short_window).mean()
        data.loc[:, 'SMA200'] = data['Close'].rolling(window=self.long_window).mean()
        if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1]:
            return 1 # Buy signal
        elif data['SMA50'].iloc[-1] < data['SMA200'].iloc[-1]:
            return -1 # Sell signal
        return 0 # Hold