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
import os
import scipy
def val_model(HED,imgs,save_path):
    def tograyImg(var):
        #var=var>0.5
        var=1-var
        image=var.data[0][0]*255
        image=image.cpu().numpy()
        image=image.astype(np.uint8)
        return image
    HED.eval()

    def crop_image(img, padding):
        h, w = img.size()[2:4]
        top_pad, bottom_pad, left_pad, right_pad = padding[0]
        crop_img = img[:, :, top_pad:h - bottom_pad, left_pad:w - right_pad]
        return crop_img
    for image_name in imgs:
        image_path = os.path.join(cfg.root_dir, image_name)
        image = io.imread(image_path)
        image = image.astype(np.float64)
        orignal_h,orignal_w,_=image.shape
        newh = image.shape[0] // 16 * 16
        neww = image.shape[1] // 16 * 16
        image = scipy.misc.imresize(image, (newh, neww))
        image = image.astype(np.float32)
        image = image[:, :, ::-1]
        image -= np.array((104.00698793, 116.66876762, 122.67891434))
        image = image.transpose((2, 0, 1))
        image = image.copy()
        image = torch.from_numpy(image)
        image=torch.unsqueeze(image,dim=0)
        image = Variable(image)
        if (cfg.gpu):
            image = image.cuda()
        c1, c2, c3, c4, c5, cfuse = HED(image)

        avg=(c1+c2+c3+c4+c5+cfuse)/6

        image1=tograyImg(c1)
        image2 = tograyImg(c2)
        image3 = tograyImg(c3)
        image4 = tograyImg(c4)
        image5 = tograyImg(c5)
        cfuse=tograyImg(cfuse)
        end=tograyImg(avg)
        min12=np.maximum(image1,image2)
        min123=np.maximum(image3,min12)
        min1234=np.maximum(image4,min123)
        min12345 = np.maximum(image5, min1234)
        min123456=np.maximum(cfuse,min12345)
        cfuse = scipy.misc.imresize(cfuse, (orignal_h, orignal_w))

        basename=os.path.basename(image_name)
        basename=basename[:-4]+'.png'
        file_name=os.path.join(save_path,basename)
        io.imsave(file_name, cfuse)

    HED.train()

if __name__ == '__main__':
    test_file_path = os.path.join(cfg.root_dir, 'test.lst')
    f = open(test_file_path)
    test_files = f.readlines()
    test_files = [x.strip('\n') for x in test_files]
    imgs = [x.split(' ')[0] for x in test_files]

    save_path='/home/mameng/deeplearning/pytorch-HED/eval/test_HED'
    HED = model.HED()
    #HED.init_weights()
    state_dict = torch.load('./snapshot/hed_epoch20.pth')#'./snapshot/hed_epoch9.pth'
    HED.load_state_dict(state_dict)

    if (cfg.gpu):
        HED = HED.cuda()
    val_model(HED,imgs,save_path)

