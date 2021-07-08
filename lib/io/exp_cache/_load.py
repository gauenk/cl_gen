def load_exp_list(exp_config_list):
    results_list = []
    for exp_config,results in zip(exp_config_list,results_list):
        results_list.append(load_exp(exp_config,order,results))
    return results_list

    return results
