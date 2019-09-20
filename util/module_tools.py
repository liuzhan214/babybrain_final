import torch 
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math 

# 输出模型参数描述
def output_name_and_params(net):
    num = 0.0
    for name, parameters in net.named_parameters():
        # if name.find('conv') != -1:
        #     print('name: {}, param: {}'.format(name, parameters.size()))
        # if name.find('upsample') != -1:
        #     print('name: {}, param: {}'.format(name, parameters.size()))
        print(name,parameters.size())
        num += parameters.numel()
    return num/1000/1000

# 计算IOU指标
def calculate_iou(img,gt_img):
    # print(img.shape,target.shape)
    class_name = ['Background','Cerebrospinal fluid','Gray matter','White matter']
    ious = {}
    for class_num in range(4):
        img_mask = (img == class_num)
        gt_mask = (gt_img == class_num)
        if (img_mask|gt_mask).sum().item() == 0:
            iou = torch.tensor(0,dtype=img.dtype)
        else:
            iou = (img_mask&gt_mask).float().sum()/(img_mask|gt_mask).float().sum()
        ious[class_name[class_num]] = iou.item()
    return ious

# 计算DSC指标
def calculate_dc(img,gt_img):
    # print(img.shape,target.shape)
    class_name = ['Background','Cerebrospinal fluid','Gray matter','White matter']
    dcs = {}
    for class_num in range(4):
        img_mask = (img == class_num)
        gt_mask = (gt_img == class_num)
        if (img_mask.float().sum() + gt_mask.float().sum()).item() == 0:
            dc = torch.tensor(1.0,dtype=img.dtype)
        else:
            dc = (img_mask&gt_mask).float().sum()*2.0/(img_mask.float().sum() + gt_mask.float().sum())
        dcs[class_name[class_num]] = dc.item()
    return dcs

# 废弃
def visualize(preds,path):
    preds = torch.argmax(preds,dim=1)
    if not os.path.exists(path):
        os.mkdir(path)
    print('start visualizion in {}...'.format(path))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # plt.imshow(preds[i,j].cpu())
            plt.savefig(os.path.join(path,'%d.png'%j))
    print('end visualizion')

# 计算accuracy指标
def calculate_acc(img,gt_img):
    # print(img.shape,target.shape)
    x = (img == gt_img).float()
    return torch.mean(x)

# cosine衰减调制学习率，废弃
def adjust_learning_rate(optimizer, epoch, epoches, initial_lr):
    lr = 0.5 * (1 + math.cos(epoch * math.pi / epoches)) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate: ', 0.5 * (1 + math.cos(epoch * math.pi / epoches)) * initial_lr)
    return lr


if __name__ == '__main__':
    import nibabel as nib
    import numpy as np
    # x1 = np.array(nib.load('/home/liuzhan/pycode/babybrain3/output/before/subject-11-label.hdr').get_data())
    # #x1 = np.array(nib.load('./data/test/subject-10-label.hdr').get_data())
    # x2 = np.array(nib.load('./output/after/subject-11-label.hdr').get_data())
    # x3 = np.array(nib.load('./data/test/subject-23-T1.hdr').get_data())
    # print(x1.shape,x2.shape,x3.shape)
    # res = calculate_dc(torch.tensor(x1),torch.tensor(x2))
    # print(res)
    # for i in range(10,24,1):
    #     x1 = np.array(nib.load('/home/liuzhan/pycode/babybrain_final/output2/subject-{}-label.hdr'.format(i)).get_data())
    #     x2 = np.array(nib.load('./data/training/subject-{}-label.hdr'.format(i)).get_data())
    #     dc = calculate_dc(torch.tensor(x1),torch.tensor(x2))
    #     print(dc)
    #     break 

    # test image hist 
    # for i in range(24,40,1):
    #     x1 = np.array(nib.load('./data/test2/subject-{}-T1.hdr'.format(i)).get_data())
    #     plt.subplot(3,6,i - 24 + 1)
    #     plt.hist(x1.flatten(),bins=5,density=True)
    #     plt.title('s{}-T1'.format(i))
    # plt.show() 
    # for i in range(11,24,1):
    #     x1 = np.array(nib.load('./data/test/subject-{}-T1.hdr'.format(i)).get_data())
    #     plt.subplot(3,5,i - 10)
    #     plt.hist(x1.flatten(),bins=5,density=True)
    #     plt.title('s{}-T1'.format(i))
    # plt.show() 