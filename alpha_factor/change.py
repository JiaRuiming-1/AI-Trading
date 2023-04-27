# only for positive estimate
for feature in alpha_field:
    alpha_df[feature] = np.where(alpha_df[feature]>=0.85, alpha_df[feature], np.where(alpha_df[feature]<=-0.85, alpha_df[feature], 0))
    alpha_df[feature] = np.where(alpha_df[feature]>=0, alpha_df[feature], 0)

# only for positive estimate
for feature in alpha_field:
    all_factors[feature] = np.where(all_factors[feature]>=0.85, all_factors[feature],
                                    np.where(all_factors[feature]<=-0.85, all_factors[feature], 0))


import cvxpy as cvx

class OptimalHoldings():
    
    def __init__(self, risk_cap=0.1, factor_max=10.0, factor_min=-10.0, weights_max=0.3, weights_min=0):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
        
    def _get_obj(self, weights, alpha_vector):
        assert (len(alpha_vector.columns) == 1)

        return cvx.Minimize(-alpha_vector.values.flatten() * weights)

    def _get_constraints(self, weights, factor_betas, risk):
        assert (len(factor_betas.shape) == 2)

        f = factor_betas.T * weights
        constraint = [risk <= self.risk_cap,
                      f <= self.factor_max, f >= self.factor_min,
                      sum(cvx.abs(weights.T)) <= 1.0,
                      weights <= self.weights_max, weights >= self.weights_min
                      ]

        return constraint

    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())
        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))

        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)

        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        #prob.solve(max_iters=100)
        prob.solve()

        optimal_weights = np.asarray(weights.value).flatten()
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)
    

dt = 20230426
obj_df = alpha_df.loc[alpha_df.trade_date==dt]
alpha_vector = obj_df.loc[obj_df.index.get_level_values(0)[-1]][['alpha_all']]
optimal_weights = OptimalHoldings().find(alpha_vector, variance_all[dt][1],
                                        variance_all[dt][4],  variance_all[dt][3])
optimal_weights.loc[optimal_weights[0]>=0.01]

## use
alpha_df['h_privious'] = 0.
positions = {}
calendar = alpha_df.trade_date.unique()

# get parameter
ticker_num = len(alpha_df.index.get_level_values(1).unique())
h0 = [0.] * ticker_num

for dt in tqdm(calendar, desc='optimized holding...'):
    # fill yesterday holding
    obj_df = alpha_df.loc[alpha_df.trade_date==dt]
    alpha_vector = obj_df.loc[obj_df.index.get_level_values(0)[-1]][['alpha_all']]
    # convex optimize
    optimal_weights = OptimalHoldings().find(alpha_vector, variance_all[dt][1],
                                        variance_all[dt][4],  variance_all[dt][3])
    h_optimal = optimal_weights
    # update optimize holding
    obj_df['h_opt'] = h_optimal.values
    obj_df['h_privious'] = h0
    positions[dt]= obj_df
    h0 = h_optimal.values
    
'''
change1 : 
  ann_days = 20
change2 ： 
  return variance, B, fr, rm.idiosyncratic_var_vector, F
  variance_all[dt] = [variance_i, B, risk_factor.iloc[-1,:], residual_i.copy(), F]
change3 :
  alpha_df['alpha_all'] = 0.3 * alpha_df['alpha_kama'] + 0.7*alpha_df['alpha_009']
change4:
  delete risk factor return calculate
change5:
  df.at[time_i,"risk.pnl"] = partial_dot_product(rr, risk_exposures[dt])
'''
