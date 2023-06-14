def concat_tests_pivot(df, PF_sig):
    """
        concat all test pins to one dataframe
        add PF flag to the last columns

        Parameters
        ----------
        df: raw df data

        PF_sig: int value
            1 is PASS, 0 is Failed

        Returns
        -------
        Date Frame :
            index : log names
            columns: pins of all test case
        dict:
            {case_name: [Pin]}
    """
    case2pins = {}
    pivote_df = pd.DataFrame()
    for df_group in df.groupby('Test_Name'):
        tmp_df = get_pivot_df_by_case(df_group)
        case2pins[df_group[0]] = df_group[1].Pin.unique()
        if pivote_df.empty:
            pivote_df = tmp_df
        pivote_df = pivote_df.merge(tmp_df, on=['log_name'], how='left')
    pivote_df['PF'] = PF_sig
    return pivote_df, case2pins

def train_data_split(df, set_size=[0.8, 0.1, 0.1], shuffle=True):
    '''
    :param df: Dateframe
        split data
    :param set_size: list(float)
        size of train valid test fraction of df
    :return:
    '''
    assert np.sum(set_size) >= 0.999 and np.sum(set_size)<=1.0
    def split_into_sets(indicies, set_size):
        last_i = 0
        sets = []

        for size in set_size:
            data_len = int(len(indicies) * size)
            sets.append(indicies[last_i:last_i+data_len])
            last_i = last_i + data_len
        return sets

    if shuffle:
        return [df.sample(frac = size) for size in set_size]
    else:
        set_indicies = split_into_sets(df.index, set_size)
        return [df.loc[indicies[0] : indicies[-1]] for indicies in set_indicies]


def rank_features_by_importance(importances, feature_names, max_print = -1):
    '''
    rank and outp features importance (attribute_name column_num entropy_value)
    :param importances: list[float]
        entropy_value list from model.feature_importances_
    :param feature_names: list[str]
        featurre name list
    :param max_print:
        print top number of features
    :return: None
    '''
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    importances_order = []
    for x_train_i in range(max_print if max_print > 0 else len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))
        importances_order.append(feature_names[indices[x_train_i]])

    return importances_order


def plot_tree_classifier(clf, tree_num = -1, feature_names=None):
    '''
    cls feature tree images expressed by matrix value of png
    :param clf: Decision Tree or RandomForest
    :param feature_names: list[str]
    :return: matrix value format by png
    '''
    if tree_num == -1:
        model = clf
    else:
        model = clf.estimators_[tree_num]

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        filled=True, rounded=True,
        special_characters=True,
        rotate=False
    )
    graph = graphviz.Source(dot_data)

    return graph


def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.2, 0.9))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()
