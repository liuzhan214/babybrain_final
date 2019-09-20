from model_dense.denseunet import DenseUNet
from losses.cross_entropy_contour import ContourCE
from util.module_tools import output_name_and_params,calculate_dc
import torch.optim as optim
from dataload import get_test_dataloader
from model_dense.denseunet import DenseUNet
from dataload import get_dataloader,get_test_dataloader
from util.module_tools import calculate_iou,calculate_dc
from config.parser import opt
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from contour import get_contour
from evaluate import evaluate_avg
import cv2
import nibabel as nib
import time
from dataload import cut_edge

label_path_arr = []
f = open('./data/test/test.txt')
lines = f.readlines()
for line in lines:
    line = line.rstrip()
    T1_path,T2_path = line.split(' ')
    label_path = T1_path.replace('T1','label')
    x = label_path.find('subject')
    label_path = label_path[x:]
    label_path_arr.append(label_path)
print(label_path_arr)

data = nib.load('./data/training/subject-1-T1.hdr')
print(data.shape)
header = data.header
affine = data.affine
# x = np.random.randint(0,4,size=(144,192,256,1),dtype=np.uint8)
# y = nib.spm2analyze.Spm2AnalyzeImage(x,affine,header)
# print(y)
# nib.save(y,'./output/fuck.hdr')

FRAMES = opt.frames
EPOCHES = opt.epoches
HEIGHT = opt.cut_height
WIDTH = opt.cut_width 
# prepare model
model = DenseUNet()
print(model)
para_num = output_name_and_params(model)
print('param num = {}'.format(para_num))
if torch.cuda.is_available():
    model.cuda()

# prepare weight storage path
save_path = './weight/densenet9_v3.pth'
#save_path = './weight/denseunet.pth'
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print('load from {}'.format(save_path))
    test_loader = get_test_dataloader()
    start_time = time.time()
    with torch.no_grad():
        step1,step2,step3 = 16,16,16
        for idx,img in enumerate(test_loader):
            D,H,W = img.shape[-3],img.shape[-2],img.shape[-1]

            img = torch.squeeze(img,dim=0)
            T1 = img[0]
            T2 = img[1]
            D_left,D_right,H_left,H_right,W_left,W_right = cut_edge(T1, keep_margin=0)
            D_right += 1 
            H_right += 1
            W_right += 1
            img = img[...,D_left:D_right,H_left:H_right,W_left:W_right]

            # check0 = img[0][img[0] > 0]
            # check1 = img[1][img[1] > 0]
            check0 = img[0]
            check1 = img[1]
            T1 = (img[0:1,...] - check0.mean())/check0.std()
            T2 = (img[1:2,...] - check1.mean())/check1.std()
            img = torch.cat([T1,T2],dim=0)
            img = torch.unsqueeze(img,dim=0)

            img = img.cuda()
            ds,de,hs,he,ws,we = D_left,D_right,H_left,H_right,W_left,W_right
            cropx,cropy = he - hs,we - ws
            
            # print(cropx,cropy) 
            pred_full = torch.zeros((1,4,de - ds,cropx,cropy),dtype=torch.float32).cuda()
            contour_pred_full = torch.zeros((1,1,de - ds,cropx,cropy),dtype=torch.float32).cuda()
            pred_cnt = torch.zeros((de - ds,cropx,cropy),dtype=torch.float32).cuda()
            
            for i in range(0,de - ds,step1):
                for j in range(0,cropx,step2):
                    for k in range(0,cropy,step3):
                        ti,tj,tk = i,j,k 
                        if i + FRAMES > de - ds:
                            ti = de - ds - FRAMES 
                        if j + HEIGHT > cropx:
                            tj = cropx - HEIGHT 
                        if k + WIDTH > cropy:
                            tk = cropy - WIDTH 
                        imgp = img[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH]
                        pred_imgp,contour_pred_imgp = model(imgp)
                        pred_full[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += pred_imgp
                        contour_pred_full[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += contour_pred_imgp
                        pred_cnt[ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += 1
            pred_full /= pred_cnt
            contour_pred_full /= pred_cnt

            pred = torch.argmax(pred_full,dim=1)
            # print(pred.shape)
            hdr_pred = pred.permute(1,2,3,0).cpu().numpy()
            hdr_pred = np.asarray(hdr_pred,dtype=np.uint8)
            full = np.zeros(shape=(D,H,W,1),dtype=np.uint8)
            # print(hdr_pred.shape,full.shape,full[:,hs:he,ws:we,:].shape)
            full[ds:de,hs:he,ws:we,:] = hdr_pred

            full = nib.spm2analyze.Spm2AnalyzeImage(full,affine,header)
            nib.save(full,'./output/{}'.format(label_path_arr[idx]))
            print('{} finish'.format(label_path_arr[idx]))
            print('debug',D,ds,de)
            for frame in range(D):
                if frame >= ds and frame < de:
                    zero = np.zeros((H,W),dtype=np.uint8)
                    x = pred[0,frame - ds].cpu().numpy()
                    zero[hs:he,ws:we] = x 
                    tmp_path = './output/{}_v'.format(label_path_arr[idx])
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    img_path = os.path.join(tmp_path,'{}.png'.format(frame))
                    plt.imsave(img_path,zero)
                    # cv2.imwrite('./output/{}/{}.png'.format(idx + 11,frame),x)
                    print('{} save finished'.format(img_path))
                else:
                    zero = np.zeros((H,W),dtype=np.uint8)
                    tmp_path = './output/{}_v'.format(label_path_arr[idx])
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    img_path = os.path.join(tmp_path,'{}.png'.format(frame))
                    plt.imsave(img_path,zero)
                    # cv2.imwrite('./output/{}/{}.png'.format(idx + 11,frame),x)
                    print('{} save finished'.format(img_path))
    end_time = time.time()
    print('time: {}'.format(end_time - start_time))
    