
def xfer_field_to_label(field):
    if field == "D": return "Patchsize, $D$"
    elif field == "std": return "Noise Level, $\sigma^2$"
    elif field == "mu2": return r"MSE($I_t$,$I_0$), $\Delta I_t$"
    elif field == "pmis": return r"Percent of Misaligned Frames"
    elif field == "ub": return r"Upper Bound of Misaligned Pixel Value"
    elif field == "T": return r"Number of Frames" 
    elif field == "est_mu2_mean": return r"Est. Mean of MSE Between Subset Aves."
    elif field == "eps": return r"Gaussian Alignment Variance"        
    else: raise ValueError(f"Uknown field [{field}]")

def translate_axis_labels(labels,logs):
    new = edict({'x':None,'y':None})
    new.x = translate_label(labels.x)
    new.y = translate_label(labels.y)
    if logs.x: new.x += " [Log Scale]"
    if logs.y: new.y += " [Log Scale]"
    return new

def xfer_fields_to_labels(fields):
    labels = []
    for field in fields:
        labels.append(xfer_field_to_label(field))
    return labels
    
    
