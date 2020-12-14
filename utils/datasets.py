import sys
import numpy as np
from scipy.io import loadmat, savemat

sys.path.append("..")

class DatasetObject:
    def __init__(self, f, ll, ld):
        self.feature = f
        self.logical_label = ll
        self.label_distribution = ld

def chlog(labels, t=0.5):
    num_ins, num_labs = labels.shape
    y_log = np.zeros_like(labels)
    for i in range(num_ins):
        y = labels[i]
        s = 0
        for _ in range(num_labs):
            j = np.argmax(y)
            s += y[j]
            labels[i, j], y_log[i,j] = 0, 1.0
            if s >= t:
                break
    return y_log

def load(name, return_X_y=True, problem='ldl', ftype=np.float32, ltype=np.float32):
    mat = loadmat("../utils/data/%s/%s.mat" % (problem, name))
    feat = mat['features'].astype(ftype)
    if return_X_y and problem == 'ldl':
        return feat, mat['logical_label'].astype(ltype), mat['label_distribution'].astype(ltype) 
    if return_X_y and problem == 'mll':
        return feat, mat['labels'].astype(ltype), None
    if (not return_X_y) and problem == 'ldl':
        return DatasetObject(feat, mat['logical_label'].astype(ltype), mat['label_distribution'].astype(ltype))
    if (not return_X_y) and problem == 'mll':
        return DatasetObject(feat, mat['labels'].astype(ltype), None)