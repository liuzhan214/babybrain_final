from dataload import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import torch 

# 四联通计算边缘，随后进行高斯模糊
def get_edge(gt_img):
    gt = gt_img.numpy().astype(dtype=np.uint8)
    m,n = gt.shape
    edge = np.zeros(shape=(m,n),dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            v = gt[i,j]
            if i + 1 < m and gt[i+1,j] != v:
                edge[i,j] = 1
            elif j + 1 < n and gt[i,j + 1] != v:
                edge[i,j] = 1
    edge = edge.astype(np.float32)
    edge = cv2.GaussianBlur(edge,(7,7),2)
    return edge 

# 计算gt_img的轮廓
def get_contour(gt_img):
    res = []
    for i in range(gt_img.shape[0]):
        arr = [] 
        for j in range(gt_img.shape[1]):
            edge = get_edge(gt_img[i,j])
            arr.append(np.expand_dims(edge,0))
        arr = np.concatenate(arr,0)
        res.append(np.expand_dims(arr,0))
    res = np.concatenate(res,0)
    return torch.tensor(res)


# if __name__ == '__main__':
#     train_loader,val_loader = get_dataloader()
#     for img,gt_img in val_loader:
#         # gt = gt_img[0,50]
#         # gt = gt.numpy().astype(dtype=np.uint8)
        
#         # m,n = gt.shape
#         # edge = np.zeros(shape=(m,n),dtype=np.uint8)
#         # for i in range(m):
#         #     for j in range(n):
#         #         v = gt[i,j]
#         #         if i + 1 < m and gt[i+1,j] != v:
#         #             edge[i,j] = 1
#         #         elif j + 1 < n and gt[i,j + 1] != v:
#         #             edge[i,j] = 1
#         # edge = edge.astype(np.float32)
#         # edge = cv2.GaussianBlur(edge,(7,7),2)
#         batch = 0
#         x = 50
#         T1_img = img[batch,0,x]
#         T2_img = img[batch,1,x]
#         gt = gt_img[batch,x]
        
#         contour = get_contour(gt_img)
#         contour_gt = contour[batch,x]


#         plt.subplot(141),plt.imshow(T1_img)
#         plt.subplot(142),plt.imshow(T2_img)
#         plt.subplot(143),plt.imshow(gt)
#         plt.subplot(144),plt.imshow(contour_gt)
#         plt.show()
#         break


