from util.module_tools import output_name_and_params,calculate_dc
from util.dataprocessing import visualize_dataset
from model_dense.denseunet import DenseUNet
from dataload import get_dataloader
from config.parser import opt
from evaluate import evaluate_avg
from contour import get_contour
from losses.cross_entropy_contour import ContourCE,SoftDiceLoss

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

FRAMES = opt.frames
HEIGHT = opt.cut_height
WIDTH = opt.cut_width 
EPOCHES = opt.epoches


if __name__ == '__main__':
    # prepare model
    model = DenseUNet()
    # print(model)
    para_num = output_name_and_params(model)
    print('param num = {}'.format(para_num))
    if torch.cuda.is_available():
        model.cuda()
    
    # prepare optimizer,lr scheduler,loss function
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=1e-2)
    scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHES,1e-6)
    loss_fn = ContourCE()
    loss_fn2 = SoftDiceLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    
    # prepare data 
    train_loader,val_loader = get_dataloader()

    # prepare metric
    best_iou = None
    # prepare weight storage path
    # save_path = './weight/denseunet_8_images.pth'
    # load_path = './weight/densenet9.pth'
    load_path = './weight/densenet9_v2_with_diceloss.pth'
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        model.eval()
        print('load from {}'.format(load_path))
        # 验证集上的估计
        val_loss,val_iou = evaluate_avg(model,val_loader,loss_fn,step1=16,step2=16,step3=16)
        best_iou = (val_iou['Cerebrospinal fluid'] + val_iou['Gray matter'] + val_iou['White matter'])/3
        print('loss = {} avg_iou = {}'.format(val_loss,best_iou))
        print('Cerebrospinal fluid:{:.3f} Gray matter:{:.3f} White matter:{:.3f}'.format(val_iou['Cerebrospinal fluid'],val_iou['Gray matter'],val_iou['White matter']))
    save_path = './weight/densenet9_v2_with_diceloss.pth'
    
    for epoch in range(EPOCHES):
        train_loss = 0
        train_ious = {}
        model.train() 
        for i,(img,gt_img) in enumerate(train_loader):
            contour = get_contour(gt_img)
            contour = contour.cuda()
            img = img.cuda()
            gt_img = gt_img.cuda()
            
            optimizer.zero_grad()
            pred,contour_pred = model(img)    
            
            if opt.contour_reg_flag == 1:
                loss = loss_fn([pred,contour_pred],[gt_img,contour]) + loss_fn2(pred,gt_img)
            else:
                loss = loss_fn(pred,gt_img)   
            train_loss += loss.item()
            iou = calculate_dc(torch.argmax(pred,dim=1),gt_img)
            for k,v in iou.items():
                if k in train_ious:
                    train_ious[k] += v 
                else:
                    train_ious[k] = v

            loss.backward()
            optimizer.step()
        scheduler_S.step(epoch)

        train_loss /= len(train_loader)
        for k in train_ious:
            train_ious[k] /= len(train_loader)
        print('train epoch {} loss = {}'.format(epoch, train_loss))
        for (k,v) in train_ious.items():
            print('{}:{:.3f}'.format(k,v),end=' ')
        print('\n')

        if epoch%opt.evaluation_interval == 0:
            val_loss,val_iou = evaluate_avg(model,val_loader,loss_fn,step1=32,step2=32,step3=32)
            print('val epoch {} loss = {}'.format(epoch,val_loss))
            print('Cerebrospinal fluid:{:.3f} Gray matter:{:.3f} White matter:{:.3f}'.format(val_iou['Cerebrospinal fluid'],val_iou['Gray matter'],val_iou['White matter']))
            avg_iou = (val_iou['Cerebrospinal fluid'] + val_iou['Gray matter'] + val_iou['White matter'])/3
            if best_iou is None:
                best_iou = avg_iou
            if avg_iou > best_iou:
                best_iou = avg_iou
                print('better found, current best iou = {}'.format(best_iou))
                torch.save(model.state_dict(), save_path)
            print('current avg val iou, avg_iou = {}'.format(avg_iou))    
            print('current best val iou, avg_iou = {}'.format(best_iou)) 
            print()

    