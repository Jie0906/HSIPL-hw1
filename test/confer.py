import numpy as np
import matplotlib.pyplot as plt
''' SAM '''
def SAM(original, target):
    x, y, z = original.shape
    
    B = (original.reshape(x * y, z)).transpose()
    
    inner_ori_target = (B * target).sum(axis=0)
    norm_ori = np.power(np.power(B, 2).sum(axis=0), 0.5)
    norm_target = np.power(np.power(target, 2).sum(axis=0), 0.5)
    x2 = inner_ori_target / (norm_ori * norm_target)   
    dr = np.arccos(abs(x2))
    
    SAM_result = 1 - dr.reshape(x, y)
    
    return SAM_result


