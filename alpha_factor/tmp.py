def rescale_value_factors(universe, zscore_dict = value_zscore_factors):
    for factor_name in tqdm(zscore_dict.keys()):
        rescale_list = zscore_dict[factor_name]
        universe[factor_name] = (universe[factor_name] - rescale_list[0])/rescale_list[1]
        universe[factor_name] = np.where(universe[factor_name].abs()>1, 
                                         np.sign(universe[factor_name]), universe[factor_name])
    return universe
        

universe = rescale_value_factors(universe)
