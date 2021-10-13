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

def rsdpw_sam(x, y, target):
    x_target_sam = SAM(x, target)
    y_target_sam = SAM(y, target)
    return max(x_target_sam/y_target_sam, y_target_sam/x_target_sam)

def rsdpw_sid(x, y, target):
    x_target_sid = SID(x, target)
    y_target_sid = SID(y, target)
    return max(x_target_sid/y_target_sid, y_target_sid/x_target_sid)

def rsdpw_sid_tan(x, y, target):
    x_target_sid_tan = SID(x, target) * np.tan(SAM(x, target))
    y_target_sid_tan = SID(y, target) * np.tan(SAM(y, target))
    return max(x_target_sid_tan/y_target_sid_tan, y_target_sid_tan/x_target_sid_tan)

def rsdpw_sid_sin(x, y, target):
    x_target_sid_sin = SID(x, target) * np.sin(SAM(x, target))
    y_target_sid_sin = SID(y, target) * np.sin(SAM(y, target))
    return max(x_target_sid_sin/y_target_sid_sin, y_target_sid_sin/x_target_sid_sin)

'''main'''

fp = r'panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'),'double')


plt.figure()
plt.title('cource practice')
plt.imshow(him[:,:,1] + (1000 * groundtruth))
plt.show()

arr = np.where(groundtruth == 1)


#print(arr)

'''(1)'''
plt.figure()
plt.title('Plot the spectral signature')
plt.xlabel('bands')
plt.ylabel('ref')
plt.plot(him[7, 37, :], label = 'p1', color = '#FF0000' ) #p1
plt.plot(him[20, 35, :], label = 'p2', color = '#FF8000' ) #p2
plt.plot(him[34, 34, :], label = 'p3', color = '#FFDC35' ) #p3
plt.plot(him[47, 33, :], label = 'p4', color = '#CA8EFF' ) #p4
plt.plot(him[59, 33, :], label = 'p5', color = '#82D900' ) #p5
plt.legend()
plt.show()

#plt.figure()
# for i in range(len(arr[0])):
#     plt.plot(him[arr[0][i],arr[1][i],:])
#     print(arr[0][i],arr[1][i])
    
''''(2)_SAM'''
print('SAM:')
SAM_p1p2 = SAM(him[7, 37, :], him[20, 35, :])
SAM_p1p3 = SAM(him[7, 37, :], him[34, 34, :])
SAM_p1p4 = SAM(him[7, 37, :], him[47, 33, :])
SAM_p1p5 = SAM(him[7, 37, :], him[59, 33, :])
print(SAM_p1p2, SAM_p1p3, SAM_p1p4, SAM_p1p5)

SAM_p2p3 = SAM(him[20, 35, :], him[34, 34, :])
SAM_p2p4 = SAM(him[20, 35, :], him[47, 33, :])
SAM_p2p5 = SAM(him[20, 35, :], him[59, 33, :])
print(SAM_p2p3, SAM_p2p4, SAM_p2p5)

SAM_p3p4 = SAM(him[34, 34, :], him[47, 33, :])
SAM_p3p5 = SAM(him[34, 34, :], him[59, 33, :])
print(SAM_p3p4, SAM_p3p5)

SAM_p4p5 = SAM(him[47, 33, :], him[59, 33, :])
print(SAM_p4p5)
print('-------------------------------------')

''''(2)_SID'''
print('SID:')
SID_p1p2 = SID(him[7, 37, :], him[20, 35, :])
SID_p1p3 = SID(him[7, 37, :], him[34, 34, :])
SID_p1p4 = SID(him[7, 37, :], him[47, 33, :])
SID_p1p5 = SID(him[7, 37, :], him[59, 33, :])
print(SID_p1p2, SID_p1p3, SID_p1p4, SID_p1p5)

SID_p2p3 = SID(him[20, 35, :], him[34, 34, :])
SID_p2p4 = SID(him[20, 35, :], him[47, 33, :])
SID_p2p5 = SID(him[20, 35, :], him[59, 33, :])
print(SID_p2p3, SID_p2p4, SID_p2p5)

SID_p3p4 = SID(him[34, 34, :], him[47, 33, :])
SID_p3p5 = SID(him[34, 34, :], him[59, 33, :])
print(SID_p3p4, SID_p3p5)

SID_p4p5 = SID(him[47, 33, :], him[59, 33, :])

print( SID_p4p5)
print('-------------------------------------')


''''(3)_SID_tan'''
print('SID_tan:')
SID_p1p2_tan = SID_p1p2 * np.tan(SAM_p1p2)
SID_p1p3_tan = SID_p1p3 * np.sin(SAM_p1p3)
SID_p1p4_tan = SID_p1p4 * np.sin(SAM_p1p4)
SID_p1p5_tan = SID_p1p5 * np.sin(SAM_p1p5)
print(SID_p1p2_tan, SID_p1p3_tan, SID_p1p4_tan, SID_p1p5_tan)


SID_p2p3_tan = SID_p2p3 * np.tan(SAM_p2p3)
SID_p2p4_tan = SID_p2p4 * np.tan(SAM_p2p4)
SID_p2p5_tan = SID_p2p5 * np.tan(SAM_p2p5)
print(SID_p2p3_tan, SID_p2p4_tan, SID_p2p5_tan)

SID_p3p4_tan = SID_p3p4 * np.tan(SAM_p3p4)
SID_p3p5_tan = SID_p3p5 * np.tan(SAM_p3p5)
print(SID_p3p4_tan, SID_p3p5_tan)


SID_p4p5_tan = SID_p4p5 * np.tan(SAM_p4p5)
print(SID_p4p5_tan)
print('-------------------------------------')

''''(3)_SID_sin'''
print('SID_sin:')
SID_p1p2_sin = SID_p1p2 * np.sin(SAM_p1p2)
SID_p1p3_sin = SID_p1p3 * np.sin(SAM_p1p3)
SID_p1p4_sin = SID_p1p4 * np.sin(SAM_p1p4)
SID_p1p5_sin = SID_p1p5 * np.sin(SAM_p1p5)
print(SID_p1p2_sin, SID_p1p3_sin, SID_p1p4_sin, SID_p1p5_sin)


SID_p2p3_sin = SID_p2p3 * np.sin(SAM_p2p3)
SID_p2p4_sin = SID_p2p4 * np.sin(SAM_p2p4)
SID_p2p5_sin = SID_p2p5 * np.sin(SAM_p2p5)
print(SID_p2p3_sin, SID_p2p4_sin, SID_p2p5_sin)


SID_p3p4_sin = SID_p3p4 * np.sin(SAM_p3p4)
SID_p3p5_sin = SID_p3p5 * np.sin(SAM_p3p5)
print(SID_p3p4_sin, SID_p3p5_sin)

SID_p4p5_sin = SID_p4p5 * np.sin(SAM_p4p5)
print(SID_p4p5_sin)
print('-------------------------------------')

''''(4)_RSDPW_SAM'''
print('RSDPW_SAM:')
RSDPW_SAM_p1p2 = rsdpw_sam(him[7, 37, :], him[20, 35, :], him[34, 34, :])
RSDPW_SAM_p1p4 = rsdpw_sam(him[7, 37, :], him[47, 33, :], him[34, 34, :])
RSDPW_SAM_p1p5 = rsdpw_sam(him[7, 37, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SAM_p1p2, RSDPW_SAM_p1p4, RSDPW_SAM_p1p5)

RSDPW_SAM_p2p4 = rsdpw_sam(him[20, 35, :], him[47, 33, :], him[34, 34, :])
RSDPW_SAM_p2p5 = rsdpw_sam(him[20, 35, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SAM_p2p4, RSDPW_SAM_p2p5)

RSDPW_SAM_p4p5 = rsdpw_sam(him[47, 33, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SAM_p4p5)
print('-------------------------------------')

''''(4)_RSDPW_SID'''
print('RSDPW_SID:')
RSDPW_SID_p1p2 = rsdpw_sid(him[7, 37, :], him[20, 35, :], him[34, 34, :])
RSDPW_SID_p1p4 = rsdpw_sid(him[7, 37, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p1p5 = rsdpw_sid(him[7, 37, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p1p2, RSDPW_SID_p1p4, RSDPW_SID_p1p5)

RSDPW_SID_p2p4 = rsdpw_sid(him[20, 35, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p2p5 = rsdpw_sid(him[20, 35, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p2p4, RSDPW_SID_p2p5)

RSDPW_SID_p4p5 = rsdpw_sid(him[47, 33, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p4p5)
print('-------------------------------------')

''''RSDPW_SID_tan'''
print('RSDPW_SID_tan:')
RSDPW_SID_p1p2_tan = rsdpw_sid_tan(him[7, 37, :], him[20, 35, :], him[34, 34, :])
RSDPW_SID_p1p4_tan = rsdpw_sid_tan(him[7, 37, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p1p5_tan = rsdpw_sid_tan(him[7, 37, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p1p2_tan, RSDPW_SID_p1p4_tan, RSDPW_SID_p1p5_tan)

RSDPW_SID_p2p4_tan = rsdpw_sid_tan(him[20, 35, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p2p5_tan = rsdpw_sid_tan(him[20, 35, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p2p4_tan, RSDPW_SID_p2p5_tan)

RSDPW_SID_p4p5_tan = rsdpw_sid_tan(him[47, 33, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p4p5_tan)
print('-------------------------------------')

''''RSDPW_SID_sin'''
print('RSDPW_SID_sin:')
RSDPW_SID_p1p2_sin = rsdpw_sid_sin(him[7, 37, :], him[20, 35, :], him[34, 34, :])
RSDPW_SID_p1p4_sin = rsdpw_sid_sin(him[7, 37, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p1p5_sin = rsdpw_sid_sin(him[7, 37, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p1p2_sin, RSDPW_SID_p1p4_sin, RSDPW_SID_p1p5_sin)

RSDPW_SID_p2p4_sin = rsdpw_sid_sin(him[20, 35, :], him[47, 33, :], him[34, 34, :])
RSDPW_SID_p2p5_sin = rsdpw_sid_sin(him[20, 35, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p2p4_sin, RSDPW_SID_p2p5_sin)

RSDPW_SID_p4p5_sin = rsdpw_sid_sin(him[47, 33, :], him[59, 33, :], him[34, 34, :])
print(RSDPW_SID_p4p5_sin)
print('-------------------------------------')