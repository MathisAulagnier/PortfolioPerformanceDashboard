import yfinance as yf

symbol = "AIR.PA"
symbol = "AAPL"
ticker = yf.Ticker(symbol)
info = ticker.info

# Afficher la 
print(info["sector"])