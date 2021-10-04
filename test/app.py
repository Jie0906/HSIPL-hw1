import numpy as np
import matplotlib.pyplot as plt


def SID(p1, p2):
    #ril = (apple.reshape(x*y,z)).transpose()
    m = p1 / (np.sum(p1,axis=0))
    n = p2 / (np.sum(p2,axis=0))
    
    D_rd = m * (np.log(m / n))
    D_rd = np.sum(D_rd,axis=0)
    D_dr = n * (np.log(n / m))
    D_dr = np.sum(D_dr,axis=0)
    
    SID_result = np.double(D_rd + D_dr)
    #SID_result = SID_rd.reshape(x,y)
    
    
    return SID_result





fp = r'panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
gt = item['groundtruth']
him = np.array(item.get('HIM'),'double')


plt.figure()
plt.imshow(him[:,:,0])


plt.figure()
plt.imshow(gt)

plt.figure('www')
plt.title('title')
plt.xlabel('x')
plt.ylabel('y')

'''for i in range(10):
    plt.plot(him[0,i,:])'''
    

arr = np.where(gt == 1)
print(arr)

for i in range(len(arr[0])):
    plt.plot(him[arr[0][i],arr[1][i],:])
    print(arr[0][i],arr[1][i])
    

a = SID(him[7,37,:],him[20, 35, :] )
print(a)