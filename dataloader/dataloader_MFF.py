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

#多聚焦融合数据loader

#~/sunyang/dataset
class MFFDataSet(Dataset):
    def __init__(self,dataset_root_dir,dataset_dict):

        self.dataset_root_dir = dataset_root_dir
        self.dataset_dict = dataset_dict
        self.WH = [2,2]
        self.HW = [2,2]
        self.far_list,self.next_list = self.get_MFF()

        #Since the size of the MFF task varies a lot, 
        #two candidate frames are set up so that the first cut scheme WH is used if W is larger than H, 
        #and HW is used if H is larger than W.
        # No distinction is needed when the cut size is set to 448*448 during training
        self.transform_WH  = TwoPathCompose([
                TwoPathRandomResizedCrop([self.WH[0]*224,self.WH[1]*224], scale=(0.64, 1.0), interpolation=3),
                TwoPathRandomHorizontalFlip(),])
        self.transform_HW  = TwoPathCompose([
                TwoPathRandomResizedCrop([self.HW[0]*224,self.HW[1]*224], scale=(0.64, 1.0), interpolation=3),
                TwoPathRandomHorizontalFlip(),])
        self.transform_same = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.win_HW = 224


    def __len__(self):
        return len(self.far_list)

    def __getitem__(self, idx):
       
        far = Image.open(self.far_list[idx]).convert('RGB')
        nxt = Image.open(self.next_list[idx]).convert('RGB')
       
        W,H = far.size
        far, nxt = self.transform_WH (far, nxt)    
        far = self.transform_same(far)
        nxt = self.transform_same(nxt)
        

        image_name = self.next_list[idx].split("/")[-1]
        train_info = {'H':self.WH[0]*self.win_HW,'W':self.WH[1]*self.win_HW,                  #image origin size
                    'H_len':self.WH[0],'W_len':self.WH[1],     #image windows size
                    'name':image_name,              #image name in dataset
           }
        #label = torch.from_numpy(np.array(label))
        return far,nxt,train_info


    def get_MFF(self):
        far_list=[]
        next_list=[]
        for name,dataset_dir in self.dataset_dict.items():
            if name=="RealMFF":
                far_dir = os.path.join(dataset_dir,"imageB")
                next_dir = os.path.join(dataset_dir,"imageA")
            elif name=="MFI-WHU":
                far_dir = os.path.join(dataset_dir,"source_2")
                next_dir = os.path.join(dataset_dir,"source_1")
            else:
                print("dataset_name Error!!!",name)
            
            
            for path in os.listdir(next_dir):
                # check if current path is a file
                if os.path.isfile(os.path.join(next_dir, path)):
                    next_list.append(os.path.join(next_dir, path))
                    if name=="RealMFF":
                        temp = path.split("_")[0]
                        temp = temp +"_B.png"
                        far_list.append(os.path.join(far_dir, temp))
                    else:
                        far_list.append(os.path.join(far_dir, path))
        
        return far_list,next_list
    


    def get_img_list(self,x):
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
    
    def recover_img(self,img_list,info):
        #print(info)
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

    def save_img(self,img_tensor,path,train_info,name):
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

