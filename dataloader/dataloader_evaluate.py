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
import torch.nn.functional as F
import torchvision.transforms as transforms
#~/sunyang/dataset
class EvaluateDataSet(Dataset):
    def __init__(self,EvalDataSet,config):

        self.EvalDataSet = EvalDataSet
        self.upsample = config['upsample']
        self.hasWindows = config['hasWindows']
        self.windows =  [448,448]
        #不同任务序号不同
        self.dataset_dict = {
            "TNO" :0,
            "LLVIP":0,
            "LLVIP_Test":0,
            "RoadScene":0,
            "MandP":0,
            "MEFB":1,
            "MEF":1,
            "SCIE_test":1,
            "Lytro":2,
            "MFF":2,
            "MFF_Win":2,
            "M3FD": 0,
        }
        
        self.rgb_list,self.t_list,self.dataset_list = self.get_RGBT()
        self.transform = None

        if self.upsample:
            self.win_HW = 112
        else:
            self.win_HW = 224
        

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        
        
        rgb = Image.open(self.rgb_list[idx]).convert('RGB')
        t = Image.open(self.t_list[idx]).convert('RGB')
        dataset = self.dataset_list[idx]

        W,H = rgb.size

        H_len = math.ceil(H/self.win_HW)
        W_len = math.ceil(W/self.win_HW)

        if self.hasWindows and W>448 and H>448:
            self.transform  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform  = transforms.Compose([
                transforms.Resize([H_len*self.win_HW,W_len*self.win_HW]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        rgb = self.transform (rgb)
        t = self.transform (t)
            
        if  self.hasWindows and (W<=448 or H<=448):
            rgb_list = rgb
            t_list = t
        elif  self.hasWindows and W>448 and H>448:
            HW_len = 448
            rgb_list = self.get_img_list(rgb,HW_len)
            t_list = self.get_img_list(t,HW_len)
        else:
            rgb_list = rgb
            t_list = t

        image_name = self.rgb_list[idx].split("/")[-1]
        dataset_name = self.dataset_list[idx]
        task_index = self.dataset_dict[self.dataset_list[idx]]
        train_info = {'H':H,'W':W,                  #image origin size
                'H_len':H_len,'W_len':W_len,     #image windows size
                'name':image_name,            #image name in dataset
                'dataset':dataset_name,      
                'task_index':task_index
        }

        return rgb_list,t_list,train_info

    def get_RGBT(self):
        res_A=[]
        res_B=[]
        res_dataset_list=[]
        for dataset_name in self.EvalDataSet.keys():
            ddir = self.EvalDataSet[dataset_name]

            if dataset_name=="LLVIP" or dataset_name=="LLVIP_Test":       
                rgb_dir = os.path.join(ddir,"visible","test")     #RGB
                t_dir = os.path.join(ddir,"infrared","test")      #t
            elif dataset_name=="MandP" or dataset_name=="M3FD":
                rgb_dir = os.path.join(ddir,"vi")     #RGB
                t_dir = os.path.join(ddir,"ir")      #t
            elif dataset_name=="MEFB" or dataset_name=="MEF" or dataset_name=="MFF" :
                rgb_dir = os.path.join(ddir,"input")             #过曝
                t_dir = os.path.join(ddir,"input")               #低曝
            elif dataset_name=="Lytro":
                rgb_dir = os.path.join(ddir,"BB")     #远焦
                t_dir = os.path.join(ddir,"AA")       #近焦
            elif dataset_name=="TNO":
                rgb_dir = os.path.join(ddir,"vi")     #RGB
                t_dir = os.path.join(ddir,"ir")       #t
            elif dataset_name=="SCIE_test":
                rgb_dir = os.path.join(ddir,"oe")     #RGB
                t_dir = os.path.join(ddir,"ue")
            else:
                print("dataset_name Error!!!",dataset_name)
            
            rgb_list=[]
            t_list=[]
            dataset_list=[]
            if dataset_name=="LLVIP" or dataset_name=="LLVIP_Test" or dataset_name=="MandP" or dataset_name=="M3FD":
                for path in os.listdir(rgb_dir):
                    # check if current path is a file
                    if os.path.isfile(os.path.join(rgb_dir, path)):
                        rgb_list.append(os.path.join(rgb_dir, path))
                        t_list.append(os.path.join(t_dir, path))
                        dataset_list.append(dataset_name)
                A_list = rgb_list
                B_list = t_list
            elif dataset_name=="MEFB" or dataset_name=="MEF" or dataset_name=="MFF": 
                for path in os.listdir(rgb_dir):
                    # check if current path is a file
                    class_path = os.path.join(rgb_dir, path)
                    if os.path.isdir(class_path):
                        for file in os.listdir(class_path):
                            file_path = os.path.join(class_path, file)
                            if "_B." in file or "_b." in file:        #B是过曝 oe    或者远焦far
                                rgb_list.append(file_path)
                               
                            if "_a." in file or "_A." in file:        #A是低曝光 ue  或者近焦next
                                t_list.append(file_path)
                                dataset_list.append(dataset_name)
                A_list = rgb_list
                B_list = t_list
                
            elif dataset_name=="Lytro":
                for path in os.listdir(rgb_dir):
                    class_path = os.path.join(rgb_dir, path)

                    if "-B." in path:
                        pathB = path.replace("-B.", "-A.")
                        classB_path = os.path.join(t_dir, pathB)
                    elif "-b." in path:
                        pathB = path.replace("-b.", "-a.")
 
                        classB_path = os.path.join(t_dir, pathB)
                    #print(class_path)
                    if os.path.isfile(class_path):
                        if "-B." in path or "-b." in path:
                            rgb_list.append(class_path)
                            t_list.append(classB_path)
                            dataset_list.append("Lytro")

                A_list = rgb_list
                B_list = t_list
            elif dataset_name=="TNO":
                for path in os.listdir(rgb_dir):
                    # check if current path is a file
                    if os.path.isfile(os.path.join(rgb_dir, path)):
                        rgb_list.append(os.path.join(rgb_dir, path))
                        t_list.append(os.path.join(t_dir, path))
                        dataset_list.append("TNO")
                A_list = rgb_list
                B_list = t_list
            elif dataset_name=="SCIE_test":
                for path in os.listdir(rgb_dir):
                    # check if current path is a file
                    if os.path.isfile(os.path.join(rgb_dir, path)):
                        rgb_list.append(os.path.join(rgb_dir, path))
                        t_list.append(os.path.join(t_dir, path))
                        dataset_list.append("SCIE_test")
                A_list = rgb_list[:12]
                B_list = t_list[:12]
                dataset_list = dataset_list[:12]
            
            print("dataset test len :",dataset_name,len(A_list),len(B_list))
            assert len(A_list)==len(dataset_list), ' dataset_list 长度不一'
            res_A = res_A + A_list
            res_B = res_B + B_list
            res_dataset_list = res_dataset_list +dataset_list
        print("res_A :",len(res_A))
        print("res_B :",len(res_A))
        assert len(res_A)==len(res_dataset_list), ' res_dataset_list 长度不一'
        
        return res_A,res_B,res_dataset_list
    

    def get_img_list(self,x,HW_len):
        img_list = self.do_get_img_list(x,0,0,HW_len)
        #print(len(img_list))
        img_list = torch.stack(img_list)
        return img_list

    def do_get_img_list(self,x,h0,w0,HW_len):
        _,H,W = x.shape
        H_len = math.ceil(H/HW_len)
        W_len = math.ceil(W/HW_len)
        img_list = []
        #print(H_len,W_len)

        for i in range(H_len):
            if i==H_len-1:
                str_H = H - HW_len - h0
                end_H = H - h0
            else:
                str_H = i*HW_len + h0
                end_H = (i+1)*HW_len + h0
            for j in range(W_len):
                if j==W_len-1:
                    str_W = W - HW_len - w0
                    end_W = W - w0
                else:
                    str_W = j*HW_len + w0
                    end_W = (j+1)*HW_len + w0
                #print(str_H,end_H,str_W,end_W)
                img_list.append(x[:,str_H:end_H,str_W:end_W])
        return img_list


    def do_recover_img(self,x,img_list,h0,w0,HW_len):
     
        H,W,=x.shape[1],x.shape[2]
        H_len = math.ceil(H/HW_len)
        W_len = math.ceil(W/HW_len)
        img = x
        for i in range(H_len):
            if i==H_len-1:
                str_H = H - HW_len -h0
                end_H = H -h0
            else:
                str_H = i*HW_len + h0
                end_H = (i+1)*HW_len + h0
            for j in range(W_len):
                if j==W_len-1:
                    str_W = W - HW_len -w0
                    end_W = W -w0
                else:
                    str_W = j*HW_len +w0
                    end_W = (j+1)*HW_len +w0
                #print(str_H,end_H,str_W,end_W)
                
                img[:,str_H:end_H,str_W:end_W] = img_list[i*W_len+j]
           
    
        return img


    
    def save_img_NewLoader(self,img_tensor_list,path,info_list):
        print(info_list)
        print(img_tensor_list.shape)
        path = os.path.join(path,info_list["dataset"])
        if not os.path.exists(path):
            os.makedirs(path)

        H = info_list["H"]
        W = info_list["W"]
        img = img_tensor_list

        x = torch.zeros(3,H,W)
        if  self.hasWindows and H>448 and W>448:
            img = self.do_recover_img(x,img,0,0,448)
        else:
            img = img.squeeze()

        re_transform = transforms.Compose([
            transforms.Resize([H,W]),
            ])
        img = re_transform(img)
        img = img.permute(1,2,0)

        img_path = os.path.join(path,info_list["name"])
        imsave(img_path, img)



