import alphalens as al
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from IPython.display import Image
from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


class NoOverlapVoter(VotingClassifier):
    def _calculate_oob_score(self, classifiers):
        oob = 0
        for clf in classifiers:
            oob = oob + clf.oob_score_
        return oob / len(classifiers)

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        estimators_ = []
        for i in range(n_skip_samples):
            estimators_.append(
                classifiers[i].fit(x[i::n_skip_samples], y[i::n_skip_samples])
            )
        return estimators_

    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf' + str(i), estimator) for i in range(n_skip_samples + 1)]

        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting=voting)

    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)

        return self


def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0

    def split_into_sets(data, set_sizes):
        last_i = 0
        sets = []
        for set_size in set_sizes:
            set_n = int(len(data) * set_size)
            sets.append(data[last_i:last_i + set_n])
            last_i = last_i + set_n

        return sets

    def split_by_index(df, sets):
        set_indicies = split_into_sets(df.index.levels[0], sets)
        return [df.loc[indicies[0]:indicies[-1]] for indicies in set_indicies]

    X_train, X_valid, X_test = split_by_index(all_x, [train_size, valid_size, test_size])
    y_train, y_valid, y_test = split_by_index(all_y, [train_size, valid_size, test_size])

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def IID_check(tmp):
    def sp(group, col1_name, col2_name):
        x = group[col1_name]
        y = group[col2_name]
        return spearmanr(x, y)[0]

    tmp['target_1'] = tmp.groupby(level=1)['return_2q'].shift(-1)
    tmp['target_2'] = tmp.groupby(level=1)['return_2q'].shift(-2)
    tmp['target_3'] = tmp.groupby(level=1)['return_2q'].shift(-3)
    tmp['target_4'] = tmp.groupby(level=1)['return_2q'].shift(-4)

    g = tmp.dropna().groupby(level=0)
    labels = ['target_1','target_2','target_3','target_4']
    for i in range(4):
        #label = 'target_'+str(i+1)
        ic = g.apply(sp, 'target', labels[i])
        ic.plot(ylim=(-1, 1), label=labels[i])
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.title('Rolling Autocorrelation of Labels Shifted Days')

def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)
    graphviz.Source(dot_data).render('output-graph.gv', view=True)
    return Image(graphviz.Source(dot_data).pipe(format='png'))


def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        #plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    for x_train_i in range(len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))


def sharpe_ratio(factor_returns, annualization_factor=np.sqrt(252)):
    return annualization_factor * factor_returns.mean() / factor_returns.std()


def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns


def plot_factor_returns(factor_returns):
    #(1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))
    (1 + factor_returns).cumprod().plot(grid=True)


def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    #ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))
    ls_FRA.plot(title="Factor Rank Autocorrelation", grid=True)


def build_factor_data(factor_data, pricing, holding_time=20):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[holding_time], max_loss=1.)
        for factor_name, data in factor_data.iteritems()}
