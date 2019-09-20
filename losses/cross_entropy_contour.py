import torch 
import torch.nn as nn 
import torch.nn.functional as F
from config.parser import opt

class ContourCE(nn.Module):
    def __init__(self):
        super(ContourCE,self).__init__()
        # opt.contour_reg_flag控制是否开启轮廓回归
        # 将背景的权重设置为1，目的是是模型更关注脊髓液，灰质和白质
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1,2,2,2],dtype=torch.float32)) 
        self.mse = nn.MSELoss()
        # alpha设置成0.1比设置成1要好，其余参数未尝试
        if opt.contour_reg_flag == 1:
            self.alpha = 0.1
        else:
            self.alpha = 0
    
    def forward(self,x,y):
        if opt.contour_reg_flag == 1:
            # 此时x1的维度为batch,4,D,H,W. y1的维度为batch,D,H,W 
            # 此时x2的维度为batch,1,D,H,W. y2的维度为batch,D,H,W 
            x1,x2 = x 
            y1,y2 = y
            x2 = torch.squeeze(x2,dim=1) 
            ce_loss = self.ce(x1,y1)
            mse_loss = self.mse(x2,y2)
            return ce_loss + self.alpha*mse_loss
        else:
            return self.ce(x,y)

# dice loss
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        # logits: b,4,D,H,W
        # targets: b D,H,W 
        one_hot = torch.eye(4,dtype=torch.int64).cuda()
        logits_size = logits.size()
        targets_size = targets.size() 

        targets = targets.flatten() 
        new_targets = one_hot[targets] 
        new_targets = torch.reshape(new_targets,targets_size + torch.Size([4]))

        score = torch.zeros(opt.batch_size,dtype=torch.float32).cuda() 
        for i in range(1,4):
            t_logits = logits[:,i,...]
            t_targets = new_targets[...,i].float()

            num = t_targets.size(0)
            smooth = 1
            probs = nn.Sigmoid()(t_logits)
            m1 = probs.view(num, -1)
            m2 = t_targets.view(num, -1)
            intersection = (m1 * m2)
    
            t_score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            t_score = 1 - t_score.sum() / num
            score += t_score
        return score.mean() 

        
