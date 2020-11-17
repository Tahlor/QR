import torch
from torch import nn

class QRCenterPixelLoss(nn.Module):
    def __init__(self,img_size,qr_size,padding,threshold=0,bigger=False):
        super(QRCenterPixelLoss, self).__init__()
        assert(qr_size<=33)
        self.threshold=threshold
        #create weight mask
        self.mask = torch.FloatTensor(img_size,img_size).zero_()
        
        cell_size = img_size/(qr_size+2*padding)
        #mask achors (with padding)
        top_left_left_x = round((padding-1)*cell_size)
        top_left_right_x = round((padding+7+1)*cell_size)
        top_left_top_y = round((padding-1)*cell_size)
        top_left_bot_y = round((padding+7+1)*cell_size)
        self.mask[top_left_top_y:top_left_bot_y,top_left_left_x:top_left_right_x]=1

        top_right_left_x = round((padding+qr_size-8)*cell_size)
        top_right_right_x = round((padding+qr_size+1)*cell_size)
        top_right_top_y = round((padding-1)*cell_size)
        top_right_bot_y = round((padding+7+1)*cell_size)
        self.mask[top_right_top_y:top_right_bot_y,top_right_left_x:top_right_right_x]=1

        bot_left_left_x = round((padding-1)*cell_size)
        bot_left_right_x = round((padding+7+1)*cell_size)
        bot_left_top_y = round((padding+qr_size-8)*cell_size)
        bot_left_bot_y = round((padding+qr_size+1)*cell_size)
        self.mask[bot_left_top_y:bot_left_bot_y,bot_left_left_x:bot_left_right_x]=1



        if qr_size>=25:
            bot_right_left_x = round((padding+qr_size-7 -2)*cell_size)
            bot_right_right_x = round((padding+qr_size-7 +3)*cell_size)
            bot_right_top_y = round((padding+qr_size-7 -2)*cell_size)
            bot_right_bot_y = round((padding+qr_size-7 +3)*cell_size)
            self.mask[bot_right_top_y:bot_right_bot_y,bot_right_left_x:bot_right_right_x]=1

        #mask pixel centers. I'll do full at very center and 0.5 around
        for cell_r in range(qr_size):
            for cell_c in range(qr_size):
                center_x = round((cell_c+padding)*cell_size + cell_size/2)
                center_y = round((cell_r+padding)*cell_size + cell_size/2)
                if self.mask[center_y,center_x]==0: #haven't masked the corner
                    if bigger:
                        self.mask[center_y-1:center_y+2,center_x-2]=0.25
                        self.mask[center_y-1:center_y+2,center_x+3]=0.25
                        self.mask[center_y-2,center_x-1:center_x+2]=0.25
                        self.mask[center_y+3,center_x-1:center_x+2]=0.25
                    self.mask[center_y-1:center_y+2,center_x-1:center_x+2]=0.5
                    self.mask[center_y,center_x]=1
        
        self.mask=self.mask[None,...] #batch dim

    def forward(self,pred,gt):
        pred = pred.mean(dim=1) #grayscale!
        assert(gt.size(1)==1)
        gt=gt[:,0]
        diff = torch.abs(pred-gt)
        diff[diff<self.threshold] = 0
        self.mask = self.mask.to(diff.device)
        diff*=self.mask
        return diff.mean()

