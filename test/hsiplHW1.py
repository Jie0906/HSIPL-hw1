import numpy as np
import matplotlib.pyplot as plt


def SAM(x, y):
    rd = np.dot(x, y)
    r_abs = (np.sum((x ** 2), axis=0)) ** 0.5 
    d_abs =(np.sum((y ** 2), axis=0)) ** 0.5
    temp = rd / (r_abs * d_abs) 
    
    SAM_result = np.arccos(temp)
    
    
    return SAM_result



def SID(x, y):
    m = x / (np.sum(x,axis=0))
    n = y / (np.sum(y,axis=0))
    
    D_rd = m * (np.log(m / n))
    D_rd = np.sum(D_rd,axis=0)
    D_dr = n * (np.log(n / m))
    D_dr = np.sum(D_dr,axis=0)
    
    SID_result = np.double(D_rd + D_dr)
    
    return SID_result



'''main'''

fp = r'panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'),'double')


plt.figure()
plt.imshow(him[:,:,1] + (1000 * groundtruth))

arr = np.where(groundtruth == 1)


#print(arr)

'''(1)'''
plt.figure()
plt.plot(him[7, 37, :]) #p1
plt.plot(him[20, 35, :]) #p2
plt.plot(him[34, 34, :]) #p3
plt.plot(him[47, 33, :]) #p4
plt.plot(him[59, 33, :]) #p5



#plt.figure()
# for i in range(len(arr[0])):
#     plt.plot(him[arr[0][i],arr[1][i],:])
#     print(arr[0][i],arr[1][i])
    
''''(2)_SAM'''
SAM_p1p2 = SAM(him[7, 37, :], him[20, 35, :])
SAM_p1p3 = SAM(him[7, 37, :], him[34, 34, :])
SAM_p1p4 = SAM(him[7, 37, :], him[47, 33, :])
SAM_p1p5 = SAM(him[7, 37, :], him[59, 33, :])
print(SAM_p1p2, SAM_p1p3, SAM_p1p4, SAM_p1p5)

SAM_p2p1 = SAM(him[20, 35, :], him[7, 37, :])
SAM_p2p3 = SAM(him[20, 35, :], him[34, 34, :])
SAM_p2p4 = SAM(him[20, 35, :], him[47, 33, :])
SAM_p2p5 = SAM(him[20, 35, :], him[59, 33, :])
print(SAM_p2p1, SAM_p2p3, SAM_p2p4, SAM_p2p5)

SAM_p3p1 = SAM(him[34, 34, :], him[7, 37, :])
SAM_p3p2 = SAM(him[34, 34, :], him[20, 35, :])
SAM_p3p4 = SAM(him[34, 34, :], him[47, 33, :])
SAM_p3p5 = SAM(him[34, 34, :], him[59, 33, :])
print(SAM_p3p1, SAM_p3p2, SAM_p3p4, SAM_p3p5)

SAM_p4p1 = SAM(him[47, 33, :], him[7, 37, :])
SAM_p4p2 = SAM(him[47, 33, :], him[20, 35, :])
SAM_p4p3 = SAM(him[47, 33, :], him[34, 34, :])
SAM_p4p5 = SAM(him[47, 33, :], him[59, 33, :])
print(SAM_p4p1, SAM_p4p2, SAM_p4p3, SAM_p4p5)

SAM_p5p1 = SAM(him[59, 33, :], him[7, 37, :])
SAM_p5p2 = SAM(him[59, 33, :], him[20, 35, :])
SAM_p5p3 = SAM(him[59, 33, :], him[34, 34, :])
SAM_p5p4 = SAM(him[59, 33, :], him[47, 33, :])
print(SAM_p5p1, SAM_p5p2, SAM_p5p3, SAM_p5p4)





''''(2)_SID'''
SID_p1p2 = SID(him[7, 37, :], him[20, 35, :])
SID_p1p3 = SID(him[7, 37, :], him[34, 34, :])
SID_p1p4 = SID(him[7, 37, :], him[47, 33, :])
SID_p1p5 = SID(him[7, 37, :], him[59, 33, :])
print(SID_p1p2, SID_p1p3, SID_p1p4, SID_p1p5)

SID_p2p1 = SID(him[20, 35, :], him[7, 37, :])
SID_p2p3 = SID(him[20, 35, :], him[34, 34, :])
SID_p2p4 = SID(him[20, 35, :], him[47, 33, :])
SID_p2p5 = SID(him[20, 35, :], him[59, 33, :])
print(SID_p2p1, SID_p2p3, SID_p2p4, SID_p2p5)

SID_p3p1 = SID(him[34, 34, :], him[7, 37, :])
SID_p3p2 = SID(him[34, 34, :], him[20, 35, :])
SID_p3p4 = SID(him[34, 34, :], him[47, 33, :])
SID_p3p5 = SID(him[34, 34, :], him[59, 33, :])
print(SID_p3p1, SID_p3p2, SID_p3p4, SID_p3p5)

SID_p4p1 = SID(him[47, 33, :], him[7, 37, :])
SID_p4p2 = SID(him[47, 33, :], him[20, 35, :])
SID_p4p3 = SID(him[47, 33, :], him[34, 34, :])
SID_p4p5 = SID(him[47, 33, :], him[59, 33, :])
print(SID_p4p1, SID_p4p2, SID_p4p3, SID_p4p5)

SID_p5p1 = SID(him[59, 33, :], him[7, 37, :])
SID_p5p2 = SID(him[59, 33, :], him[20, 35, :])
SID_p5p3 = SID(him[59, 33, :], him[34, 34, :])
SID_p5p4 = SID(him[59, 33, :], him[47, 33, :])
print(SID_p5p1, SID_p5p2, SID_p5p3, SID_p5p4)