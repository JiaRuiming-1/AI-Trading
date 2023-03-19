## Project Instrcution
In this project wo practice last step of quant trading. The project well implements all steps what I learned from make up factor to backtesting.

## Example Demo
`Backtesting.ipynb` is a simple example of backtesting. This demo step is:
1. construct two stock based on sine and cosine data.
2. make up some alpha factors as we knew it does work, like sine and cosine drivate functions.
3. calculate portfolio by PCA
4. optimize holding by `scipy.optimize.fmin_l_bfgs_b`
5. evaluate exposure of factor and costs
6. arrtibute profit and losts to factors costs and risk
