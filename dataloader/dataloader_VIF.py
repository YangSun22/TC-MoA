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


class RGBTDataSet(Dataset):
    """ RGBT dataset
 
    :dataset_root_dir: Root directory of the RGBT dataset
    :upsample: Whether to perform upsampling images within the network X2
    :dataset_dict: Dictionary storing names and paths of VIF task datasets
    :rgb_list: list of rgb images
    :t_list: list of t images
    :arbitrary_input_size: Whether the images inside the dataset are dynamic in size or not
    """
    def __init__(self,dataset_root_dir,dataset_dict,upsample=False,arbitrary_input_size=True):

        self.dataset_root_dir = dataset_root_dir
        self.upsample =upsample
        self.dataset_dict = dataset_dict   #dict type
        self.rgb_list,self.t_list = self.get_RGBT()
       
        self.transform_normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
   
        self.arbitrary_input_size = arbitrary_input_size
        if self.upsample:
            self.win_HW = 112
        else:
            self.win_HW = 224
        
    def __len__(self):
        return len(self.rgb_list)


    def __getitem__(self, idx):
        #load multi-source images
        rgb = Image.open(self.rgb_list[idx]).convert('RGB')
        t = Image.open(self.t_list[idx]).convert('RGB')

        #load arbitrary input or not. if not arbitrary input,model only do with fixed size input
        if self.arbitrary_input_size:
            W,H = rgb.size
        else:
            #fixed size input, size must bigger than (224,224) and size==N*self.win_HW N>=1
            W,H = (448,448)  

        # images have how many shift windows in ViT
        H_len = math.ceil(H/self.win_HW)
        W_len = math.ceil(W/self.win_HW)

        #Data Augmentation
        self.transform_crop  = TwoPathCompose([
                TwoPathRandomResizedCrop([H_len*self.win_HW,W_len*self.win_HW], scale=(0.81, 1.0), interpolation=3),
                TwoPathRandomHorizontalFlip(),])
        rgb,t = self.transform_crop(rgb,t)
        rgb = self.transform_normalize(rgb)
        t= self.transform_normalize(t)
        
        image_name = self.rgb_list[idx].split("/")[-1]
        #Some info important or useful during training
        train_info = {'H':H,'W':W,                  #image origin size
                    'H_len':H_len,'W_len':W_len,     #image windows size
                    'name':image_name,              #image name in dataset
            }

        return rgb,t,train_info

    #
    def get_RGBT(self):
        """ imports each dataset in dataset_dict sequentially
            Returns a list of sample paths for each modality
        """
        rgb_list=[]
        t_list=[]
        for name,dataset_dir in self.dataset_dict.items():
            if name=="M3FD":
                rgb_dir = os.path.join(dataset_dir,"vi")
                t_dir = os.path.join(dataset_dir,"ir")
            elif name=="M3FD_DET":
                rgb_dir = os.path.join(dataset_dir,"vi")
                t_dir = os.path.join(dataset_dir,"ir")
            elif name=="LLVIP":
                rgb_dir = os.path.join(dataset_dir,"visible","train")
                t_dir = os.path.join(dataset_dir,"infrared","train")
            elif name=="MSRS":
                rgb_dir = os.path.join(dataset_dir,"train","vi")
                t_dir = os.path.join(dataset_dir,"train","ir")
                # rgb_dir = 2*rgb_dir
                # t_dir = 2*t_dir
            else:
                print("dataset_name Error!!!",self.dataset_dict)

            for path in os.listdir(rgb_dir):
                # check if current path is a file
                if os.path.isfile(os.path.join(rgb_dir, path)):
                    rgb_list.append(os.path.join(rgb_dir, path))
                    t_list.append(os.path.join(t_dir, path))

        # rgb_list = 2*rgb_list
        # t_list = 2*t_list
        
        return rgb_list,t_list
    


    def get_img_list(self,x):
        """ Cut the input tensor by window size 
            input (3,H,W)
            Return tensor for winows list (N,3,win_HW,win_HW)
        """
        _,H,W = x.shape
        win_HW = self.win_HW
        H_len = math.ceil(H/win_HW)
        W_len = math.ceil(W/win_HW)
        #print(H,W,H_len,W_len)

        img_list = []

        for i in range(H_len):
            if i==H_len-1:
                str_H = H - win_HW
                end_H = H
            else:
                str_H = i*win_HW
                end_H = (i+1)*win_HW
            for j in range(W_len):
                if j==W_len-1:
                    str_W = W - win_HW
                    end_W = W
                else:
                    str_W = j*win_HW
                    end_W = (j+1)*win_HW
                #print(str_H,end_H,str_W,end_W)
                img_list.append(x[:,str_H:end_H,str_W:end_W])
       # print(len(img_list))
        img_list = torch.stack(img_list)
        return img_list
    
    def recover_img(self,img_list,train_info):
        """ Recover the tensor of the winows list into a single image tensor.
            input (N,3,win_HW,win_HW)
            return (3,H,W)
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
        #print(train_info)
        H,W = train_info['H'][0].item(),train_info['W'][0].item()

        if not os.path.exists(path):
            os.makedirs(path)
        #print(img_tensor.shape)
        #img = img_tensor.permute(2,0,1)
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




