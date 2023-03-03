import pandas as pd
import  numpy as np
import cvxpy as cvx
from abc import ABC, abstractmethod


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):

        raise NotImplementedError()

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):

        raise NotImplementedError()

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
        prob.solve(max_iters=500)
        # prob.solve()

        optimal_weights = np.asarray(weights.value).flatten()
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)


class OptimalHoldings(AbstractOptimalHoldings):
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

    def __init__(self, risk_cap=0.1, factor_max=10.0, factor_min=-10.0, weights_max=0.65, weights_min=-0.65):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min


class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function
        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector
        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert (len(alpha_vector.columns) == 1)

        return cvx.Minimize(alpha_vector.values.T * weights + self.lambda_reg * cvx.norm(weights))

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55,
                 weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function
        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector
        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        x_star = (alpha_vector - alpha_vector.mean()) / np.abs(alpha_vector).sum()
        return cvx.Minimize(cvx.norm(weights - x_star.values.flatten()))