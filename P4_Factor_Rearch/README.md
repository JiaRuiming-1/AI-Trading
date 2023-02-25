## Project Instruction

## Get Date
  We load data from [Tushare](https://tushare.pro/) platform, which is a quant trading data supplier and most of data can be use for free.
  We process data and add some indicators by `stockstats` python package.
  Here is an example of coding in `Tushare_Coding.ipynb` file you can get a view.

## review math concept
If we got some factors relate to our portfolio risk and return, we can bulid model to calculate that.

We can construct return model r = Bf + s, each variance represent a matrix. r=Return, B=exposure of factor, s=can't explain variance. This model just explained as a liner model.

Then, we calculate volatility of portfolio by facors denote E(rrT)

<img src="images/1.jpg" width="500px"><img src="images/2.jpg" width="500px">

Most of time, if we got our factors, we may seperate factor matrix as alpha matrix and risk matrix. Because we don't want to contraint our alpha facor in convex optimization process.
