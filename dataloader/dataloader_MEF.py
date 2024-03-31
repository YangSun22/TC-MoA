import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageFilter
import cv2
from util import *
from  torchvision import utils as vutils
from scipy.misc import  imsave
import cv2 as cv
import math
from util.TwoPath_transforms import *
import torchvision.transforms as transforms




#多曝光融合数据loader

#~/sunyang/dataset
class MEFDataSet(Dataset):
    def __init__(self,dataset_root_dir,dataset_dict):

        self.dataset_root_dir = dataset_root_dir
       
        self.dataset_dict = dataset_dict
        # MEF images are oversized so only a fixed size region is taken for training (448,448)
        # In order to enhance the model Learning the interaction between picture windows   Set the fixed window size to 448, 448 
        self.H_len = 2
        self.W_len = 2
        self.win_HW = 224
        self.oe_list,self.ue_list = self.get_MEF()
       
        self.transform = TwoPathCompose([
                TwoPathRandomResizedCrop([self.H_len*self.win_HW,self.W_len*self.win_HW], scale=(0.2, 1.0), interpolation=3),
                TwoPathRandomHorizontalFlip(),])
        self.transform_same = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __len__(self):
        return len(self.oe_list)

    def __getitem__(self, idx):

        
        oe = Image.open(self.oe_list[idx]).convert('RGB')
        ue = Image.open(self.ue_list[idx]).convert('RGB')
        W,H = oe.size
       
       
        oe,ue = self.transform(oe,ue)
        oe = self.transform_same (oe)
        ue = self.transform_same (ue)
        
        image_name = self.oe_list[idx].split("/")[-1]
        train_info = {'H':self.H_len*self.win_HW,'W':self.W_len*self.win_HW,                  #image origin size
                    'H_len':self.H_len,'W_len':self.W_len,     #image windows size
                    'name':image_name,              #image name in dataset
           }


        return oe,ue,train_info


    def get_MEF(self):
        oe_list=[]
        ue_list=[]
        
        for name,dataset_dir in self.dataset_dict.items():
            if name=="SCIE":
                oe_dir = os.path.join(dataset_dir,"oe")
                ue_dir = os.path.join(dataset_dir,"ue")
            else:
                print("dataset_name Error!!!",self.dataset_dict)
            
            
            for path in os.listdir(oe_dir):
                # check if current path is a file
                if os.path.isfile(os.path.join(oe_dir, path)):
                    oe_list.append(os.path.join(oe_dir, path))
                    ue_list.append(os.path.join(ue_dir, path))
        
        return oe_list,ue_list
    


    def get_img_list(self,x):
        """ Cut the input tensor by window size 
            input (3,H,W)
            Return tensor for winows list (N,3,win_HW,win_HW)
        """
        _,H,W = x.shape
        H_len = math.ceil(H/self.win_HW)
        W_len = math.ceil(W/self.win_HW)
        #print(H,W,H_len,W_len)

        img_list = []

        for i in range(H_len):
            if i==H_len-1:
                str_H = H - self.win_HW
                end_H = H
            else:
                str_H = i*self.win_HW
                end_H = (i+1)*self.win_HW
            for j in range(W_len):
                if j==W_len-1:
                    str_W = W - self.win_HW
                    end_W = W
                else:
                    str_W = j*self.win_HW
                    end_W = (j+1)*self.win_HW
                img_list.append(x[:,str_H:end_H,str_W:end_W])
        img_list = torch.stack(img_list)
        return img_list
    
    def recover_img(self,img_list,train_info):
        """ Recover the tensor of the winows list into a single image tensor.
        input (N,3,win_HW,win_HW)
        """
        win_HW = self.win_HW
        H_len,W_len = train_info['H_len'],train_info['W_len']
        resize_H = H_len*win_HW
        resize_W = W_len*win_HW

        img = torch.zeros(3, resize_H,resize_W)
        for i in range(H_len):
            if i==H_len-1:
                str_H = resize_H - win_HW
                end_H = resize_H
            else:
                str_H = i*win_HW
                end_H = (i+1)*win_HW
            for j in range(W_len):
                if j==W_len-1:
                    str_W = resize_W - win_HW
                    end_W = resize_W
                else:
                    str_W = j*win_HW
                    end_W = (j+1)*win_HW
                img[:,str_H:end_H,str_W:end_W] = img_list[i*W_len+j]
       # img = img.permute(1,2,0)
        return img

    def save_img(self,img_tensor,path,train_info,name=None):
        """ Save an image tensor to a specified location

        """
        H,W = train_info['H'][0].item(),train_info['W'][0].item()
        if not os.path.exists(path):
            os.makedirs(path)
        re_transform = transforms.Compose([
            transforms.Resize([H,W]),
            ])
        img = re_transform(img_tensor)
        img = img.permute(1,2,0)

        if name!=None:
            img_path = os.path.join(path,name)
        else:
            img_path = os.path.join(path,train_info['name'])
        imsave(img_path, img)

import torchvision.transforms as transforms



