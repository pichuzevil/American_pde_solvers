import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class MarketDataLoader:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        
    def get_risk_free_rate(self):
        """Fetches the 3-Month Treasury Yield as a decimal."""
        try:
            tnx = yf.Ticker("^IRX")
            hist = tnx.history(period="1d")
            return hist['Close'].iloc[-1] / 100
        except Exception:
            return 0.045  # Standard fallback

    def get_historical_volatility(self, days=252):
        """Calculates annualized historical volatility as a fallback."""
        hist = self.ticker.history(period=f"{days}d")
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        return log_returns.std() * np.sqrt(252)

    def get_option_parameters(self, target_expiry_idx=2):
        # 1. Spot Price
        spot = self.ticker.history(period="1d")['Close'].iloc[-1]
        
        # 2. Expiry and Time to Maturity (T)
        expiries = self.ticker.options
        chosen_expiry = expiries[target_expiry_idx]
        expiry_dt = datetime.strptime(chosen_expiry, '%Y-%m-%d')
        T = max((expiry_dt - datetime.now()).days / 365.0, 1/365.0)
        
        # 3. Dividend Yield (q) logic
        raw_q = self.ticker.info.get('dividendYield', 0.0)
        if raw_q is not None:
            q = raw_q / 100 if raw_q > 0.20 else raw_q
        else:
            q = 0.0
        
        # 4. Option Chain and Volatility Check
        chain = self.ticker.option_chain(chosen_expiry)
        puts = chain.puts
        idx = (puts['strike'] - spot).abs().idxmin()
        target_put = puts.loc[idx]
        
        # --- VOLATILITY FALLBACK LOGIC ---
        sigma = target_put['impliedVolatility']
        
        # If IV is missing or zero, we use historical vol, otherwise we use a default floor
        if sigma < 0.0001:
            print(f"Warning: IV is 0.0% for {self.ticker_symbol}. Using historical volatility.")
            sigma = self.get_historical_volatility()
        
        # Final floor to ensure numerical stability in the PDE
        sigma = max(sigma, 0.05) 
        
        return {
            'S0': spot,
            'K': target_put['strike'],
            'T': T,
            'sigma': sigma,
            'r': self.get_risk_free_rate(),
            'q': q,
            'market_price': target_put['lastPrice']
        }