# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys 
from typing import Iterable
import os
import torch
import time
import util.misc as misc
import util.lr_sched as lr_sched
from util.ema import EMA



def train_one_iter(model,task_index,batch_A,batch_B,device,global_rank,iter_num,optimizer,loss_scaler,ema,train_info,config):
    samples_A = batch_A.to(device, non_blocking=True)
    samples_B = batch_B.to(device, non_blocking=True)

    batch_size,_,_,_ = samples_A.shape
    #Train an iter and backpropagate the gradient
    with torch.cuda.amp.autocast():
        loss_dict, pred,att_tuple= model(samples_A, samples_B ,task_index,iter_num[0])
        loss = loss_dict["loss"]  
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                update_grad=True)
    if ema!=None:
        name_list=ema.getname()
        state_dict = model.state_dict()
        for name in name_list:
            param_new = ema(name, state_dict[name])
            state_dict[name] = param_new

    optimizer.zero_grad()
    #Save the generated fusion images at regular intervals
    if iter_num[-1]%config["save_img_interval"]==0 and global_rank==0:
        B,C,_,_ = pred.shape
        pred = pred[0].detach()
    else:
        pred = None
    
    torch.cuda.synchronize()

    return loss_dict,pred
        

def train_one_epoch(model: torch.nn.Module,
                    data_loader_list, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    config=None,
                    global_rank=None,
                    ema = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    

    loss_chunk_list=[]
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_type = None
    task_dict= {"VIF":0,"MEF":1,"MFF":2}
 
    #The outermost loop requires a fixed data_loader，The default is the first task VIF
    if config["VIF"]:  
        if data_loader_type==None:
            data_loader_type="VIF"
            LoopUsed_loader = data_loader_list["VIF"]
            LoopUsed_dataset = LoopUsed_loader.dataset
        else:
            VIF_data_loader = data_loader_list["VIF"]
            VIF_data_loader_iter =  iter(VIF_data_loader)
            VIF_dataset = VIF_data_loader.dataset
    if config["MEF"]: 
        if data_loader_type==None:
            data_loader_type="MEF"
            LoopUsed_loader = data_loader_list["MEF"]
            LoopUsed_dataset = LoopUsed_loader.dataset
        else:
            MEF_data_loader = data_loader_list["MEF"]
            MEF_data_loader_iter =  iter(MEF_data_loader)
            MEF_dataset = MEF_data_loader.dataset
    if config["MFF"]:
        if data_loader_type==None:
            data_loader_type="MFF"
            LoopUsed_loader = data_loader_list["MFF"]
            LoopUsed_dataset = LoopUsed_loader.dataset
        else:
            MFF_data_loader = data_loader_list["MFF"]
            MFF_data_loader_iter =  iter(MFF_data_loader)
            MFF_dataset = MFF_data_loader.dataset

    
    #Import ema module
    if config["use_ema"]:
        name_list = ema.getname()

    time_start = time.time()
    for data_iter_step, (samples_rgb, samples_t,rgb_train_info) in enumerate(metric_logger.log_every(LoopUsed_loader, print_freq, header)):
        exist_loss_dict = {}
        # Load all the datasets for the tasks that exist
        if config["MEF"] and data_loader_type!="MEF":
            try:
                    samples_oe,samples_ue,MEF_train_info = MEF_data_loader_iter.next()
            except:
                    MEF_data_loader_iter = iter(MEF_data_loader)
                    samples_oe,samples_ue,MEF_train_info = MEF_data_loader_iter.next()
        
        if config["MFF"] and data_loader_type!="MFF":
            try:
                    samples_far,samples_nxt,MFF_train_info = MFF_data_loader_iter.next() 
            except:
                    MFF_data_loader_iter = iter(MFF_data_loader)
                    samples_far,samples_nxt,MFF_train_info = MFF_data_loader_iter.next()

        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(LoopUsed_loader) + epoch, config)
        
        #Training a VIF iter
        if config["VIF"]:
            loss_RGBT,pred= train_one_iter(model,task_dict["VIF"],samples_rgb,samples_t,device,global_rank,[epoch,data_iter_step],optimizer,loss_scaler,ema,rgb_train_info,config)
            exist_loss_dict["RGBT"] = loss_RGBT
            if pred!=None:
                LoopUsed_dataset.save_img(pred.cpu(),os.path.join(config["output_img_dir"],config["method_name"],"VIF"),rgb_train_info,
                                    str(epoch)+"_"+str(data_iter_step/config["save_img_interval"])+".jpg"
                                    )
            if config["use_ema"]:
                state_dict = model.state_dict()
                for name in name_list:
                    param_new = ema(name, state_dict[name])
        #Training a MEF iter
        if config["MEF"] and data_loader_type!="MEF":
            loss_MEF,pred = train_one_iter(model,task_dict["MEF"],samples_oe,samples_ue,device,global_rank,[epoch,data_iter_step],optimizer,loss_scaler,ema,MEF_train_info,config)
            exist_loss_dict["MEF"] = loss_MEF
            if pred!=None:
                #fusion = MEF_dataset.recover_img(pred,MEF_info)
                MEF_dataset.save_img(pred.cpu(),os.path.join(config["output_img_dir"],config["method_name"],"MEF"),MEF_train_info,
                                    str(epoch)+"_"+str(data_iter_step/config["save_img_interval"])+".jpg",
                                    )
            if config["use_ema"]:
                state_dict = model.state_dict()
                for name in name_list:
                    param_new = ema(name, state_dict[name])
        #Training a MFF iter
        if config["MFF"] and data_loader_type!="MFF":
            loss_MFF,pred = train_one_iter(model,task_dict["MFF"],samples_far,samples_nxt,device,global_rank,[epoch,data_iter_step],optimizer,loss_scaler,ema,MFF_train_info,config)
            exist_loss_dict["MFF"] = loss_MFF
            if pred!=None:
                #fusion = MFF_dataset.recover_img(pred,MFF_info)
                MFF_dataset.save_img(pred.cpu(),os.path.join(config["output_img_dir"],config["method_name"],"MFF"),MFF_train_info,
                                    str(epoch)+"_"+str(data_iter_step/config["save_img_interval"])+".jpg"
                                    )
            if config["use_ema"]:
                state_dict = model.state_dict()
                for name in name_list:
                    param_new = ema(name, state_dict[name])

        # Record each loss
        loss =  None    
        for key in exist_loss_dict.keys():
            if loss ==None:
                loss = exist_loss_dict[key]["loss"]
            else:
                loss = loss + exist_loss_dict[key]["loss"]
          
        loss_value = loss.item()

        if  torch.isnan(loss ):
            print(" loss_nan", loss)
        if not math.isfinite(loss_value):
            print("Loss is {}, should stopping training".format(loss_value))
            continue
            #sys.exit(1)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None :
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(LoopUsed_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            #Export all loss values to tenserboard
            for loss_key in exist_loss_dict.keys():
                for key in exist_loss_dict[loss_key].keys():
                    if key!='loss':
                        temp = exist_loss_dict[loss_key][key].item()
                        log_writer.add_scalar(loss_key+"_"+key, temp, epoch_1000x)
        
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

