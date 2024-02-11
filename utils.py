import torch
import numpy as np

def _normalize(table, dim, nans=False):
    table_corrected = table.nan_to_num() if nans else table
    return table / table_corrected.sum(dim, keepdim=True).expand(table.size())

def _fix_nan(table):
    n_infs = torch.Tensor([-np.inf]).expand(table.size())
    return torch.where(table.isnan(), n_infs, table)