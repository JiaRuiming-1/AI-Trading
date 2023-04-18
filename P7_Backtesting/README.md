## Project Instrcution
In this project wo practice last step of quant trading. The project well implements all steps what I learned from make up factor to backtesting.

## Example Demo
`Backtesting_animate.ipynb` is a simple example of backtesting. This demo step is:
1. construct two stock based on sine and cosine data.
2. make up some alpha factors as we knew it does work, like sine and cosine drivate functions.
3. calculate portfolio by PCA
4. optimize holding by `scipy.optimize.fmin_l_bfgs_b`
5. evaluate exposure of factor and costs
6. arrtibute profit and losts to factors costs and risk

## Alpha Factors
`Factor_Process.ipynb` this notebook construct some alpha factors 
make some alpha factors based on papers and Alpha191 and evaluate them.

## ML learn to trade
`RandomForest_Combine_Factor.ipynb` combine all alpha facotrs by RadomForest ML method.

## BackTest
`Backtest_Real.ipynb` this notebook is backtesting by real-world data
1. Optimal portfolio bettwen ML combined factor and PCA risk model by convex method.
2. Analysis risk expousure ,alpha expousure and costs
3. Analysis Pnl(profit and loss) and factor attributes.
