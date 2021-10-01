import numpy as np
import matplotlib.pyplot as plt
''' SAM '''
def SAM(original, target):
    x, y, z = original.shape#取得照片寬度，高度，維度
    
    B = (original.reshape(x * y, z)).transpose()#將矩陣轉置
    
    inner_ori_target = (B * target).sum(axis=0)#將基準向量與待測點向量內稽
    norm_ori = np.power(np.power(B, 2).sum(axis=0), 0.5)#計算基準值的向量 
    norm_target = np.power(np.power(target, 2).sum(axis=0), 0.5)#計算待測點的向量
    x2 = inner_ori_target / (norm_ori * norm_target)#公式( <r*d>/|r|*|d| ) =x2   
    dr = np.arccos(abs(x2))#將數值arccos(x2)
    
    SAM_result = 1 - dr.reshape(x, y)#將取得的數值轉成相片矩陣
    
    return SAM_result


