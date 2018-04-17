import model
from data.BSDdataset import BSDdata
from config import cfg
from torch.utils import data
import torch.optim as optim
from utils.losses import CrossEntropyLoss
import torch
import numpy as np
import skimage.io as io
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
def train_model(HED,optimzer,traindata_loader):

    def adjust_lr(optimzer):
        for param_group in optimzer.param_groups:
            param_group['lr']=param_group['lr']*0.1
    def show_lr(optimzer):
        for param_group in optimzer.param_groups:
            print('lr:{}'.format(param_group['lr']))
    def crop_image(img,padding):
        h,w=img.size()[2:4]
        top_pad,bottom_pad,left_pad,right_pad=padding[0]
        crop_img=img[:,:,top_pad:h-bottom_pad,left_pad:w-right_pad]
        return img

    def crop(d):
        d_h, d_w = d.size()[2:4]
        g_h, g_w = d_h - 8, d_w - 8
        d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
             int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
        return d
    def toImage(image):
        image=image[0]
        image=image.data.cpu().numpy()
        image = image.transpose((1, 2,0))
        image += np.array((122.67892, 116.66877, 104.00699))
        image=image.astype(np.uint8)
        return image

    def tograyImg(var):
        image = var.data[0][0] * 255
        image = image.cpu().numpy()
        image = image.astype(np.uint8)
        return image
    loss_sum,loss1_sum,loss2_sum,loss3_sum,loss4_sum,loss5_sum,loss6_sum=(0,0,0,0,0,0,0)
    for epoch in range(1,cfg.max_epoch):
        HED.train()
        batch_count=0
        optimzer.zero_grad()
        for ii,(image,label,padding) in enumerate(traindata_loader):
            image=Variable(image)
            label=Variable(label)
            """ 
            padding = padding.numpy()
            #image=crop_image(image,padding)
            label=crop_image(label,padding)
            #image_show=toImage(image)
            label_show=tograyImg(label)
            #io.imshow(image_show)
            #plt.show()
            io.imshow(label_show)
            plt.show()
            """
            batch_count=batch_count+1
            padding=padding.numpy()
            if(cfg.gpu):
                image=image.cuda()
                label=label.cuda()
            c1,c2,c3,c4,c5,cfuse=HED(image)
            label=crop(crop_image(label,padding))
            loss1=criterion(crop(crop_image(c1,padding)),label)
            loss2=criterion(crop(crop_image(c2,padding)),label)
            loss3=criterion(crop(crop_image(c3,padding)),label)
            loss4 = criterion(crop(crop_image(c4,padding)), label)
            loss5 = criterion(crop(crop_image(c5,padding)), label)
            loss6 = criterion(crop(crop_image(cfuse,padding)), label)
            loss=loss1+loss2+loss3+loss4+loss5+loss6
            loss_sum=loss_sum+loss.data[0]
            loss1_sum = loss1_sum + loss1.data[0]
            loss2_sum = loss2_sum + loss2.data[0]
            loss3_sum = loss3_sum + loss3.data[0]
            loss4_sum = loss4_sum + loss4.data[0]
            loss5_sum = loss5_sum + loss5.data[0]
            loss6_sum = loss6_sum + loss6.data[0]
            loss.backward()
            #torch.nn.utils.clip_grad_norm(HED.parameters(),max_norm=5.0)
            if(ii%cfg.display==0):

                show_lr(optimzer)
                print('loss:{} loss1:{} loss2:{} loss3:{} loss4:{} loss5:{} loss6:{}'.format(
                    loss_sum/cfg.display,loss1_sum/cfg.display
                    ,loss2_sum/cfg.display,loss3_sum/cfg.display
                    ,loss4_sum/cfg.display,loss5_sum/cfg.display,
                    loss6_sum/cfg.display
                ))
                loss_sum, loss1_sum, loss2_sum, loss3_sum, loss4_sum, loss5_sum, loss6_sum = (0, 0, 0, 0, 0, 0, 0)
            if(batch_count%cfg.iter_size==0):
                optimzer.step()
                optimzer.zero_grad()
                batch_count=0
        if(epoch %cfg.eopch_update_lr==0):
            adjust_lr(optimzer)
        model_save_path='{}/hed_epoch{}.pth'.format(cfg.model_save_path,epoch)
        torch.save(HED.state_dict(),model_save_path)

if __name__ == '__main__':
    traindataset = BSDdata(cfg.root_dir)
    traindata_loader = data.DataLoader(traindataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1)
    valdataset=BSDdata(cfg.root_dir)
    valdata_loader = data.DataLoader(valdataset, batch_size=1, shuffle=True, num_workers=1)

    HED = model.HED()
    #HED.init_weights()
    state_dict = torch.load('vgg16_convert.pth')#'./snapshot/hed_epoch9.pth'
    HED.load_state_dict(state_dict)

    HED.init_weights()
    ddd=HED.state_dict()
    criterion = CrossEntropyLoss()

    side_param_key_weight=['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight']#lr:cfg.lr*0.01
    side_param_key_bias=['conv1.bias','conv2.bias','conv3.bias','conv4.bias','conv5.bias']
    side_param_weight=[param for name,param in HED.named_parameters() if name in side_param_key_weight]
    side_param_bias = [param for name, param in HED.named_parameters() if name in side_param_key_bias]
    fuse_param_key_weight='convfuse.weight'
    fuse_param_weight=[param for name,param in HED.named_parameters() if name in fuse_param_key_weight]
    fuse_param_key_bias='convfuse.bias'
    fuse_param_bias=[param for name,param in HED.named_parameters() if name in fuse_param_key_bias]


    base1_4stage_param=[]
    base5satge_param=[]
    for name,param in HED.basevgg.named_parameters():
        if('conv5' in name):
            base5satge_param.append(param)
        else:
            base1_4stage_param.append(param)


    optimzer=optim.Adam([{'params':side_param_weight,'lr':cfg.lr,'weight_decay':cfg.weight_decay},
        {'params': side_param_bias, 'lr': cfg.lr, 'weight_decay': 0},
        {'params':fuse_param_weight,'lr':cfg.lr*0.1,'weight_decay':cfg.weight_decay},
        {'params': fuse_param_bias, 'lr': cfg.lr * 0.2, 'weight_decay': 0},
        {'params': base5satge_param, 'lr': cfg.lr * 5},
        {'params': base1_4stage_param}
    ],lr=cfg.lr,eps=1e-8)
    """ 
    optimzer = optim.SGD([
        {'params':side_param_weight,'lr':cfg.lr*0.01,'weight_decay':cfg.weight_decay},
        {'params': side_param_bias, 'lr': cfg.lr * 0.02, 'weight_decay': 0},
        {'params':fuse_param_weight,'lr':cfg.lr*0.001,'weight_decay':cfg.weight_decay},
        {'params': fuse_param_bias, 'lr': cfg.lr * 0.002, 'weight_decay': 0},
        #{'params': convTranspose_weight, 'lr': cfg.lr * 0, 'weight_decay': cfg.weight_decay},
        {'params': base5satge_param, 'lr': cfg.lr * 100},
        {'params': base1_4stage_param}
    ], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    """
    if (cfg.gpu):
        HED = HED.cuda()
    train_model(HED,optimzer,traindata_loader)

