import torch
import numpy as np

"""

# -----------------------
# --     USAGE 1       --
# -----------------------

if append = False
we "cat" a list of results across the dims
dims = { fieldname: dimension, ...}

We use "dims" because we want to:

(a) cat results 
    for each batch in the batch dim 

(b) cat results across
    different batches 

# -----------------------
# --     USAGE 2       --
# -----------------------

if append = True
we "append" the inputs to a list of output results


# -----------------------
# --     More Info     --
# -----------------------

we share the code between the two conceptual functions
because the nesting of if statements to access the list
is the code complexity we are interested in sharing.

"""

def format_tensor_results(results_input,results_output,dims,append=True):
    """
    results_i (dict): results input
    { "pixel_diff": {"scores":[...],"scores_t":[...],...},
      "cog_v1": {"scores":[...],"scores_t":[...],...},
      "bss": [...],
      ... }
    """
    for metric_group in results_input.keys():
        # -- select dimension --
        if metric_group in dims.keys(): dim = dims[metric_group]
        else: dim = dims['default']
        # print("metric_group: ",metric_group)

        # -- select format func --
        mgroup = results_input[metric_group]
        if isinstance(mgroup,dict): 
            format_tensor_dict(metric_group,results_input,results_output,dim,append)
        elif isinstance(mgroup,list) or isinstance(mgroup,np.ndarray): 
            format_tensor_list(metric_group,results_input,results_output,dim,append)
        else:
            raise TypeError(f"Uknown metric group type [{type(mgroup)}]")

def format_tensor_list(metric_group,results_input,results_output,dim,append=True):
    # -- Note: metric is a misnomer in this function --

    # -- init metric group if not already --
    if not(metric_group in results_output.keys()): results_output[metric_group] = []

    # -- group together into output result --
    metric = results_input[metric_group]
    # print(results_output[metric_group])
    # print("metric: ",metric)
    # print("metric[0]: ",metric[0])

    # -- cat together potential list --
    if isinstance(metric,list):
        if not torch.is_tensor(metric[0]):
            if isinstance(metric[0],list) or isinstance(metric[0],np.ndarray):
                metric = [np.array(m) for m in metric]                
                # metric = torch.cat(metric,dim=dim)
                metric = np.concatenate(metric,axis=dim)
    if isinstance(metric,list) and isinstance(metric[0],np.ndarray):
        # metric = torch.cat(metric,dim=dim)
        metric = np.concatenate(metric,axis=dim)

    # -- append result if necessary --
    if append: results_output[metric_group].append(metric)
    else: results_output[metric_group] = metric

def format_tensor_dict(metric_group,results_input,results_output,dim,append=True):

    # -- init metric group if not already --
    if not(metric_group in results_output.keys()):
        results_output[metric_group] = {}
        for metric_name in results_input[metric_group].keys():
            results_output[metric_group][metric_name] = []

    # -- group together into output result --
    for metric_name in results_input[metric_group].keys():
        metric = results_input[metric_group][metric_name]
        # -- cat together potential list --
        if isinstance(metric,list):
            # metric = torch.cat(metric,dim=dim)
            metric = np.concatenate(metric,axis=dim)
        # -- append result if necessary --
        if append: results_output[metric_group][metric_name].append(metric)
        else: results_output[metric_group][metric_name] = metric


# -----------------------------------
#
#         Stack Records
#
# -----------------------------------

def stack_ndarray(records,dims):

    for metric_group in records.keys():
        # -- select dimension --
        if metric_group in dims.keys(): dim = dims[metric_group]
        else: dim = dims['default']

        # -- select format func --
        mgroup = records[metric_group]
        if isinstance(mgroup,dict): 
            stack_ndarray_dict(metric_group,records,dim)
        elif isinstance(mgroup,list): 
            stack_ndarray_list(metric_group,records,dim)
        else:
            raise TypeError(f"Uknown metric group type [{type(mgroup)}]")

def stack_ndarray_list(metric_group,records,dim):
    metric = records[metric_group]
    metric = np.stack(metric,axis=dim)
    records[metric_group] = metric
    
def stack_ndarray_dict(metric_group,records,dim):
    for metric_name in records[metric_group].keys():
        metric = records[metric_group][metric_name]
        metric = np.stack(metric,axis=dim)
        records[metric_group][metric_name] = metric

