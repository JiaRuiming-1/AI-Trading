In thes project we utilize a beakout strategy based on Herd Behavior. So the destination is find out high and low position for long and short operation.
After that, we analysis if log return be nomal. If not, why? As momentum strategy, I need to find out outline tikers and trade ticker which be more normal. 

What I learned from this project:

1. Plot position line by pd.rolling(window)
2. Mark long and shor positon in table and on graph
3. Calculate strategy log return and plot histogram graph
4. Find outline ticker by Kolmogorov-Smirnov test (KS test) and implementation through `from scipy.stats import kstest` and `kstest(data['signal_return'], 'norm', (signal_returns_mean, signal_returns_std))`
5. Use `Ks_value > ks_threshold and p_value < p_threashold` to find outline tickers
