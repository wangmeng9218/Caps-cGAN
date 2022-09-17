import torch
import torch.nn as nn
from torch.autograd import Variable


def loss_builder(loss_type, multitask,class_num=4):
    if loss_type == 'mix_dice':
        criterion_1 = nn.CrossEntropyLoss(weight=None, ignore_index=255)
        criterion_2 = DiceLoss(class_num=class_num)
        criterion_4 = nn.MSELoss()
        if multitask == True:
            criterion_3 = nn.BCELoss()
    if loss_type in ['mix_dice', 'mix_eldice']:
        criterion_1.cuda()
        criterion_2.cuda()
        criterion = [criterion_1, criterion_2]
        if multitask == True:
            criterion_3.cuda()
            criterion.append(criterion_3)
        criterion.append(criterion_4)

    return criterion

class DiceLoss(nn.Module):
    '''
    如果某一类不存在，返回该类dice=1，dice_loss正常计算（平均值可能会偏高）
    如果所有类都不存在，返回dice_loss=0
    理论上smooth应该用不到，正常计算存在的某类dice时，union一定不为0
    '''
    def __init__(self, class_num=4,smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,input, target):
        input = torch.exp(input) #网络输出log_softmax
        self.smooth = 0.
        #torch.Tensor((3))生成一个size=3的随机张量，torch.Tensor([3])生成一个size=1值为3的张量
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]    #[12,512,256]
            target_i = (target == i).float()    #[12,512,256]
            intersect = (input_i*target_i).sum()      #无广播
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        return dice_loss



class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):

        input = input[:,0]
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()

        intersection = (input * target.float()).sum()
        unionset = input.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (unionset + self.smooth)
        #print(dice)
        Dice += dice

        dice_loss = 1 - Dice
        return dice_loss

class DiceLossUn(nn.Module):
    '''
    如果某一类不存在，返回该类dice=1，dice_loss正常计算（平均值可能会偏高）
    如果所有类都不存在，返回dice_loss=0
    理论上smooth应该用不到，正常计算存在的某类dice时，union一定不为0
    '''
    def __init__(self, class_num=2,smooth=1e-8):
        super(DiceLossUn, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.Dice_Un = SoftDiceLoss().cuda()

    def forward(self, input, target, Un):
        self.smooth = 0.
        predict = torch.argmax(input, dim=1)  # predicts预测结果nchw,寻找通道维度上的最大值predict变为nhw
        Real_GT = predict==target

        Un_GT = 1 - Real_GT.long()

        dice_un = self.Dice_Un(Un,Un_GT)

        #torch.Tensor((3))生成一个size=3的随机张量，torch.Tensor([3])生成一个size=1值为3的张量
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]    #[12,512,256]
            target_i = (target == i).float()    #[12,512,256]
            intersect = (input_i*target_i).sum()      #无广播
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        Dice_Score_Total = dice_loss+dice_un

        return Dice_Score_Total









class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=1,gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss

palette = [0, 1, 2, 3]
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        class_map=(mask==colour)
        semantic_map.append(class_map)
    semantic_map = torch.cat(semantic_map, dim=1)
    semantic_map = semantic_map.int()
    return semantic_map
