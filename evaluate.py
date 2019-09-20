from model_dense.denseunet import DenseUNet
from dataload import get_dataloader
from util.module_tools import calculate_iou,calculate_dc
from config.parser import opt
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from contour import get_contour
from dataload import cut_edge

# 16,32,32 used for quick evaluate
# 16,16,16 used for precise evaluate
# 8,8,8 used for final test 
def evaluate_avg(model,val_loader,loss_fn,step1=32,step2=32,step3=32):
    FRAMES = opt.frames
    HEIGHT = opt.cut_height
    WIDTH = opt.cut_width 
    EPOCHES = opt.epoches

    with torch.no_grad():
        model.eval()
        losses = 0
        ious = {}
        itr = 0
        for idx,(img,gt_img) in enumerate(val_loader):
            contour = get_contour(gt_img)
            contour = contour.cuda()
            img = img.cuda()
            gt_img = gt_img.cuda()
            D,H,W = img.shape[-3],img.shape[-2],img.shape[-1]
            pred_full = torch.zeros((1,4,D,H,W),dtype=torch.float32).cuda()
            contour_pred_full = torch.zeros((1,1,D,H,W),dtype=torch.float32).cuda()
            pred_cnt = torch.zeros((D,H,W),dtype=torch.float32).cuda()
            for i in range(0,D,step1):
                for j in range(0,H,step2):
                    for k in range(0,W,step3):
                        ti,tj,tk = i,j,k 
                        if i + FRAMES > D:
                            ti = D - FRAMES 
                        if j + HEIGHT > H:
                            tj = H - HEIGHT 
                        if k + WIDTH > W:
                            tk = W - WIDTH 
                        imgp = img[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH]
                        pred_imgp,contour_pred_imgp = model(imgp)
                        pred_full[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += pred_imgp
                        contour_pred_full[:,:,ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += contour_pred_imgp
                        pred_cnt[ti:ti+FRAMES,tj:tj+HEIGHT,tk:tk+WIDTH] += 1

            # print((pred_cnt == 0).nonzero())
            pred_full /= pred_cnt
            contour_pred_full /= pred_cnt
            
            if opt.contour_reg_flag == 1:
                loss = loss_fn([pred_full,contour_pred_full],[gt_img,contour])
            else:
                loss = loss_fn(pred_full,gt_img)
            
            losses += loss 
            iou = calculate_dc(torch.argmax(pred_full,dim=1),gt_img)
            
            for k,v in iou.items():
                if k in ious:
                    ious[k] += v 
                else:
                    ious[k] = v
        losses = losses/len(val_loader)
        for k in ious:
            ious[k] = ious[k]/len(val_loader)
        return losses,ious


if __name__ == '__main__':
    from model_dense.denseunet import DenseUNet
    from losses.cross_entropy_contour import ContourCE
    from util.module_tools import output_name_and_params,calculate_dc
    import torch.optim as optim
    from dataload import get_test_dataloader
    # prepare model
    model = DenseUNet()
    print(model)
    para_num = output_name_and_params(model)
    print('param num = {}'.format(para_num))
    if torch.cuda.is_available():
        model.cuda()
    

    # prepare optimizer,lr scheduler,loss function
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=1e-2)
    scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.epoches,1e-6)
    loss_fn = ContourCE()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    
    # prepare data 
    test_loader = get_test_dataloader()
    train_loader,val_loader = get_dataloader()
    

    # prepare weight storage path
    save_path = './weight/densenet9_test2.0.pth'
    #save_path = './weight/denseunet.pth'
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        model.eval()
        print('load from {}'.format(save_path))
        test_loss,test_iou = evaluate_avg(model,val_loader,loss_fn,step1=16,step2=16,step3=16)
        best_iou = (test_iou['Cerebrospinal fluid'] + test_iou['Gray matter'] + test_iou['White matter'])/3
        print('loss = {} avg_iou = {}'.format(test_loss,best_iou))
        print('Cerebrospinal fluid:{:.3f} Gray matter:{:.3f} White matter:{:.3f}'.format(test_iou['Cerebrospinal fluid'],test_iou['Gray matter'],test_iou['White matter']))


    
