import sys
import os
import requests

import torch
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from einops import rearrange, reduce, repeat
import time
from dataloader.dataloader_evaluate import *
import datetime as dt
import time
import torch.nn.functional as F
import yaml
import argparse
import model.ViT_MAE as VIT_MAE_Origin
from torchvision.utils import save_image

# define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('TC-MoA', add_help=False)
    parser.add_argument('--config_path', default='config/predict.yaml', type=str,
                        help='config_path to load')
    return parser

    

def test_one_iter(model,A,B,task_index,config):
    A = A.to(device, non_blocking=True)
    B = B.to(device, non_blocking=True)
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)

    with torch.cuda.amp.autocast():
        loss_dict, pred,att_tuple= model(A, B ,task_index)
        loss = loss_dict["loss"]

    return loss_dict,pred,att_tuple

# load an image
def main(output_dir,model,config):
    #model.eval()
    
    Evaluate_dataset = EvaluateDataSet(config['EvalDataSet'],config)
    model = model.eval()
  
    with torch.no_grad():

        i = 0
        time_start = time.time()
        for item in Evaluate_dataset:
                A,B,AB_info =item
                if i%100 == 0:
                    print("has done img num:", i)
                loss_AB,pred,_= test_one_iter(model,A,B,AB_info["task_index"],config)  
                Evaluate_dataset.save_img_NewLoader(pred.cpu(),output_dir,
                                        AB_info)
                i+=1
        time_end = time.time()
        time_sum = time_end - time_start  
        print("Time Used:", time_sum)
        
    print("Done!")

def modelTrans(ckp):
    #print(ckp['model'].keys())
    ckp = ckp['model']
    new_ckp = {}
    for key in ckp.keys():
        newkey = key
        if "blocks_SSF" in key:
            newkey = newkey.replace("blocks_SSF_rgb","blocks_MoA")
            if "mlp" in key:
                newkey = newkey.replace("mlp","MoA")
            if "SE_en_linear" in key:
                newkey = newkey.replace("SE_en_linear","dimReduction")
            if "SSF_beta" in key:
                newkey = newkey.replace("SSF_beta","modal_shifts")
        if "en_projAdapter" in key:
             newkey = newkey.replace("en_projAdapter","FusionLayer")
        if "de_projAdapter" in key:
             newkey = newkey.replace("de_projAdapter","de_FusionLayer")
        

        new_ckp[newkey] = ckp[key]
    print(new_ckp.keys())
    return new_ckp


        


def prepare_model(model_select,chkpt_dir, config,oldckp):
    # build model
    arch = config["model_type"]
    if model_select == "Base":
        print("model_type:   Base")
        models_mae =  VIT_MAE_Origin
    
    model = getattr(models_mae, arch)(config).to(config["device"])
    # load model
    checkpoint = torch.load(chkpt_dir)
    if oldckp:
        checkpoint = modelTrans(checkpoint)
        msg = model.load_state_dict(checkpoint, strict=False)
    else:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    with open(args.config_path, 'r') as stream: 
        config = yaml.safe_load(stream)

    waiting_time = config["waiting_time"]
    upsample = True
    oldckp=True

    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Waiting Hours: " ,waiting_time)
    time.sleep(waiting_time*3600)
    device=config['device']

    for ckpt_name,model_select in config["ckpt_dict"].items():
        
        model_mae = prepare_model(model_select,os.path.join(config["chkpt_dir"],ckpt_name), config,oldckp).to(device)
        print('Model loaded.',device)


        MoreDetail = config["more_detail"]
        model_type = ckpt_name +"_"+ MoreDetail
        print("model_type:",model_type)
    
        output_dir = os.path.join(config["result_path"],model_type)
        main(output_dir,model_mae,config)


