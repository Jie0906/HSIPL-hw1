import numpy as np
import matplotlib.pyplot as plt
fp = r'panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
gt = item['groundtruth']
him = np.array(item.get('HIM'),'double')


plt.figure()
plt.imshow(him[:,:,100])


plt.figure()
plt.imshow(gt)

# plt.figure('www')
# plt.title('title')
# plt.xlabel('x')
# plt.ylabel('y')

# '''for i in range(10):
#     plt.plot(him[0,i,:])'''
    

# arr = np.where(gt == 1)
# print(arr)

# for i in range(len(arr[0])):
#     plt.plot(him[arr[0][i],arr[1][i],:])
    # print(arr[0][i],arr[1][i])