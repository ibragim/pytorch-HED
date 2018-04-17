from torch.utils import data
import os
from skimage import io,transform
from config import cfg
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy
class BSDdata(data.Dataset):
    def __init__(self,root_dir):
        self.root=root_dir
        train_file_path = os.path.join(self.root, 'train_pair.lst')
        f = open(train_file_path)
        train_files = f.readlines()
        train_files = [x.strip('\n') for x in train_files]
        self.imgs = [x.split(' ')[0] for x in train_files]
        self.train_label = [x.split(' ')[1] for x in train_files]
    def __getitem__(self, index):
        image_path=os.path.join(self.root,self.imgs[index])
        label_apth=os.path.join(self.root,self.train_label[index])

        image,label,padding=self.get_image_label(image_path,label_apth)
        image = image.astype(np.float32)
        image = image[:, :, ::-1]
        image -= np.array((104.00698793, 116.66876762, 122.67891434))
        image=image.transpose((2,0,1))
        image=image.copy()
        label = label / 255.0
        label = label > 0.0
        label = label.astype(np.float32)
        label=np.expand_dims(label,axis=0)
        image=torch.from_numpy(image)
        label=torch.from_numpy(label)
        padding=torch.from_numpy(np.array(padding))
        return image,label,padding
    def __len__(self):
        return len(self.imgs)

    def get_image_label(self,image_path,label_path):
        image=io.imread(image_path)
        label=io.imread(label_path)
        image=image.astype(np.float64)
        if len(label.shape) == 3:
            label01=np.bitwise_or(label[:, :, 0],label[:, :, 1])
            label=np.bitwise_or(label01,label[:, :, 2])
        label = label.astype(np.float64)
        newh = image.shape[0] // 16 * 16
        neww = image.shape[1] // 16 * 16
        image = scipy.misc.imresize(image, (newh, neww))
        label = scipy.misc.imresize(label, (newh, neww))
        """ 
        h,w,_=image.shape
        scale=cfg.IMAGE_MIN_SIZE/min(h,w)
        max_dim=max(h,w)
        if(scale*max_dim>cfg.IMAGE_MAX_SIZE):
            scale=cfg.IMAGE_MAX_SIZE/max_dim
        image=scipy.misc.imresize(image,(round(scale*h),round(scale*w)))
        h,w,_=image.shape
        left_pad=(cfg.IMAGE_MAX_SIZE-w)//2
        right_pad=cfg.IMAGE_MAX_SIZE-w-left_pad
        top_pad=(cfg.IMAGE_MAX_SIZE-h)//2
        bottom_pad=cfg.IMAGE_MAX_SIZE-h-top_pad
        padding=[(top_pad,bottom_pad),(right_pad,left_pad),(0,0)]
        image=np.pad(image,padding,mode='constant')

        h,w=label.shape
        label=scipy.misc.imresize(label,(round(scale*h),round(scale*w)))
        padding=[(top_pad,bottom_pad),(left_pad,right_pad)]
        label=np.pad(label,padding,mode='constant')
        #h, w = label.shape
        #label=label[top_pad:h-bottom_pad,right_pad:w-left_pad]
        padding=(top_pad,bottom_pad,left_pad,right_pad)
        """
        padding=(0,0,0,0)
        return image,label,padding


