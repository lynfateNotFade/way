import sys
import numpy as np
from scipy.io import loadmat

sys.path.append("..")

class DatasetObject:
    def __init__(self, f, ll, ld):
        self.feat = f
        self.logical_label = ll
        self.label_distribution = ld

def load(name, return_X_y=False, problem='ldl', ftype=np.float32, ltype=np.float32):
    mat = loadmat("../utils/data/%s/%s.mat" % (problem, name))
    feat = mat['features'].astype(ftype)
    label = mat['labels'].astype(ltype)
    if return_X_y and problem == 'ldl':
        return feat, None, label
    elif return_X_y and problem == 'mll':
        return feat, label, None
    elif (not return_X_y) and problem == 'ldl':
        return DatasetObject(feat, None, label)
    else:
        return DatasetObject(feat, label, None)
