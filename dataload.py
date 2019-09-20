import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms.functional as F
import cv2
from config.parser import opt 

# 制作数据集类
class BabyBrain(Dataset):
    def __init__(self,discription_path,transform=None,target_transform=None,co_transform=None):
        super(BabyBrain,self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        f = open(discription_path)
        lines = f.readlines()
        self.T1imgs,self.T2imgs,self.gt_imgs = [],[],[]
        for line in lines:
            line = line.rstrip('\n')
            lines = line.split(' ')
            if len(lines) == 3:
                T1,T2,gt = lines
                self.T1imgs.append(T1)
                self.T2imgs.append(T2)
                self.gt_imgs.append(gt)
            elif len(lines) == 2:
                T1,T2 = lines
                self.T1imgs.append(T1)
                self.T2imgs.append(T2)
        f.close()
        
        
    def __getitem__(self, index):
        T1img_path = self.T1imgs[index]
        T2img_path = self.T2imgs[index]
        gt_img = None
        if len(self.gt_imgs) > 0:
            gt_img_path = self.gt_imgs[index]
        T1img = np.array(nib.load(T1img_path).get_data())
        T2img = np.array(nib.load(T2img_path).get_data())
        if len(self.gt_imgs) > 0:
            gt_img = np.array(nib.load(gt_img_path).get_data())
        img = np.concatenate([T1img,T2img],axis=-1)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            gt_img = self.target_transform(gt_img)
        if self.co_transform:
            img,gt_img = self.co_transform((img,gt_img))
        if len(self.gt_imgs) > 0:
            return (img, gt_img)
        else:
            return img
 
    def __len__(self):
        return len(self.T1imgs)

# 通用Transform类
class Transform(object):
    def __init__(self,trans:list):
        super(Transform,self).__init__()
        self.trans = trans
    
    def __call__(self, inputs):
        outputs = inputs
        for tran in self.trans:
            outputs = tran(outputs)
        return outputs    

# 3D裁剪策略，裁剪出非背景区域
def cut_edge(data, keep_margin):
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1
    eps = 1e-2
    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)


def img_totensor(img):
    img = np.transpose(img,axes=(3,0,1,2))
    return torch.tensor(img,dtype=torch.float32) 
    
def gt_img_totensor(gt_img):
    gt_img = np.squeeze(gt_img,axis=-1)
    return torch.tensor(gt_img,dtype=torch.int64) 


def co_roi_crop(img_gt_img):
    img,gt_img = img_gt_img
    T1 = img[0]
    T2 = img[1]
    ds,de,hs,he,ws,we = cut_edge(T1, keep_margin=0)
    img,gt_img = img[...,ds:de + 1,hs:he + 1,ws:we + 1],gt_img[...,ds:de + 1,hs:he + 1,ws:we + 1]
    return img,gt_img 

def co_random_crop(img_gt_img):
    img,gt_img = img_gt_img
    FRAMES = opt.frames
    HEIGHT = opt.cut_height
    WIDTH = opt.cut_width 
    i = np.random.randint(0,img.shape[-3] - FRAMES + 1)
    j = np.random.randint(0,img.shape[-2] - HEIGHT + 1)
    k = np.random.randint(0,img.shape[-1] - WIDTH + 1)
    return img[...,i:i+FRAMES,j:j+HEIGHT,k:k+WIDTH],gt_img[...,i:i+FRAMES,j:j+HEIGHT,k:k+WIDTH]

def co_standard(img_gt_img):
    img,gt_img = img_gt_img 
    check0 = img[0]
    check1 = img[1]
    # check0 = img[0][img[0] > 0]
    # check1 = img[1][img[1] > 0]
    T1 = (img[0:1,...] - check0.mean())/check0.std()
    T2 = (img[1:2,...] - check1.mean())/check1.std()
    img = torch.cat([T1,T2],dim=0)
    return img,gt_img 


def get_dataloader():
    transform = Transform([img_totensor])
    target_transform = Transform([gt_img_totensor])
    co_transform = Transform([co_roi_crop,co_standard,co_random_crop])
    co_transform_val = Transform([co_roi_crop,co_standard])
    train_set = BabyBrain('./data/training/train.txt',transform=transform,target_transform=target_transform,co_transform=co_transform)
    val_set = BabyBrain('./data/training/val.txt',transform=transform,target_transform=target_transform,co_transform=co_transform_val)
    train_loader = DataLoader(train_set,batch_size=opt.batch_size,num_workers=0,shuffle=False)
    val_loader = DataLoader(val_set,batch_size=1,num_workers=0,shuffle=False)
    return train_loader,val_loader

def get_test_dataloader():
    transform = Transform([img_totensor])
    test_set = BabyBrain('./data/test/test.txt',transform=transform,target_transform=None,co_transform=None)
    test_loader = DataLoader(test_set,batch_size=1,num_workers=0,shuffle=False)
    return test_loader

def get_test_dataloader2():
    transform = Transform([img_totensor])
    test_set = BabyBrain('./data/test2/test2.txt',transform=transform,target_transform=None,co_transform=None)
    test_loader = DataLoader(test_set,batch_size=1,num_workers=0,shuffle=False)
    return test_loader

def calcalate_mean_std():
    mean = []
    std = []
    train_loader,val_loader = get_dataloader()
    for i,(img,gt_img) in enumerate(val_loader):
        T1img = img[:,0,...].flatten()
        T2img = img[:,1,...].flatten()
        mean.append(T1img)
        std.append(T1img)
    mean = torch.cat(mean,dim=0)
    std = torch.cat(std,dim=0)
    print(mean.mean())
    print(std.std())

if __name__ == '__main__':
    train_loader,val_loader = get_dataloader()
    for img,gt_img in train_loader:
        print(img.shape,gt_img.shape)
        print(img.mean(),img.std())
    
    print('----------------------')
    
    for img,gt_img in val_loader:
        print(img.shape,gt_img.shape)
        print(img.mean(),img.std())
    
    # test_loader = get_test_dataloader() 
    # for img in test_loader:
    #     print(img.shape)
    
    # test_loader2 = get_test_dataloader2()
    # for img in test_loader2:
    #     print(img.shape)