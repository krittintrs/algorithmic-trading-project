from agent_super import TradingAgent
class RSIAgent(TradingAgent):
    def __init__(self, window=14, initial_cash=100000):
        super().__init__(initial_cash)
        self.name="RSI"
        self.window = window

    def calculate_rsi(self, data):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        data = data.copy()  # Avoid SettingWithCopyWarning
        data['RSI'] = self.calculate_rsi(data)

        if data['RSI'].iloc[-1] < 30:
            return 1  # Buy signal
        elif data['RSI'].iloc[-1] > 70:
            return 2  # Sell signal
        return 0  # Hold