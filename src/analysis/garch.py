import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Load stock data (for example, AAPL closing prices)
data = pd.read_csv('data/raw/stock_data/AAPL_stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Calculate daily returns
data['Returns'] = data['Close'].pct_change().dropna()

# Fit GARCH(1,1) model
garch_model = arch_model(data['Returns'][1:], vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Print summary
print(garch_fit.summary())

# Get volatility predictions (conditional volatility)
garch_volatility = garch_fit.conditional_volatility

# Plot volatility
plt.figure(figsize=(10, 6))
plt.plot(garch_volatility, label='GARCH Volatility')
plt.title('GARCH Model Estimated Volatility')
plt.legend()
plt.show()

# Save volatility to a CSV file
data['GARCH_Volatility'] = garch_volatility
data.to_csv('data/raw/heston_garch/garch_volatility_output.csv', index=True)

