# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import  Attention,Mlp,DropPath
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from util.pos_embed import get_2d_sincos_pos_embed
from util.fusion_loss import *
import math
from util.mefssim import MEF_MSSSIM
from model.Windows_Shift import Relative_Position_Layer,Block,window_partition,window_reverse
from model.TC_MoA import ConvFusionLayer,BiMixtureOfAdapters


def recover_Norm(image):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    imagenet_mean= imagenet_mean.view([3,1,1])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    imagenet_std = imagenet_std .view([3,1,1])
    return torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).float()


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self,  patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = (patch_size,patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x,(H,W)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, config=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.interval_tau = config['interval_tau']
        self.task_num = config['task_num']
        self.drop_path = 0
        self.tau_shift_value = config['tau_shift_value']
        self.embed_dim = embed_dim
        self.upsample = config['upsample']
        self.warmup_epochs = config['warmup_epochs']

        self.patch_embed = PatchEmbed( patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.shift_window_shape = config['shift_window_size']

        # --------------------------------------------------------------------------
        #Encoder
        self.en_relative_position = nn.ModuleList([
            Relative_Position_Layer(self.shift_window_shape, num_heads)
            for i in range(depth//2)])
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer,drop_path = self.drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        #Encoder TC-MoA
        self.blocks_MoA = nn.ModuleList([
            BiMixtureOfAdapters(embed_dim,32,self.task_num)
            for i in range(depth//self.interval_tau)])
        self.FusionLayer =  nn.ModuleList([
                ConvFusionLayer(embed_dim,32)
                for i in range(depth//self.interval_tau)])

        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1,197, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.de_relative_position = nn.ModuleList([
            Relative_Position_Layer(self.shift_window_shape, decoder_num_heads)
            for i in range(decoder_depth//2)])
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer,drop_path = self.drop_path)
            for i in range(decoder_depth)])

        # Decoder TC-MoA
        self.decoder_blocks_MoA = nn.ModuleList([
            BiMixtureOfAdapters(decoder_embed_dim,32,self.task_num)
            for i in range(decoder_depth//self.interval_tau)])
        self.de_FusionLayer =  nn.ModuleList([
                ConvFusionLayer(decoder_embed_dim,32)
                for i in range(decoder_depth//self.interval_tau)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        #define Loss
        self.mse = torch.nn.MSELoss()
        self.MaxGradLoss = MaxGradLoss(3) 
        #when input img just one, PixelLoss=MaxPixelLoss
        self.PixelLoss = PixelLoss(3)
        self.MaxPixelLoss = MaxPixelLoss(1)
        self.MFFselect = MaxGradTokenSelect()
        self.msssim_loss = SSIM()
        self.MEFSSIM = MEF_MSSSIM(is_lum=True)

        #define main branch and adapters branch balance
        init_values = 0.5
        alpha_en_len = len(self.blocks_MoA)
        alpha_de_len = len(self.decoder_blocks_MoA)
        self.Alpha_encoder = nn.Parameter(init_values * torch.ones((alpha_en_len)), requires_grad=True)
        self.Alpha_decoder = nn.Parameter(init_values * torch.ones((alpha_de_len)), requires_grad=True)
        
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 14, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 14, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x,H,W):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = H
        w = W
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs


    def forward_encoder(self, x,t, task_index):
        # embed patches
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            t = F.interpolate(t, scale_factor=2, mode='bilinear')
        
        x,(H,W) = self.patch_embed(x)
        t,(H,W) = self.patch_embed(t)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :].view(1,14,14,-1).permute(0,3,1,2)
        pos_embed = F.interpolate(pos_embed,size=(int(W),int(H)),mode='bilinear')
        pos_embed = pos_embed.view(1,-1,W*H).permute(0,2,1)

        x = x + pos_embed
        t = t + pos_embed

        window_size=self.shift_window_shape
        shift_size=7
        B,L,C = x.shape

        aux_loss = 0

        x_prompt_list=[]
        t_prompt_list=[]

        # apply Transformer blocks
        for i,blk in enumerate(self.blocks):
            have_shift = False
            if i % 2 ==1:
                have_shift = True

            x = x.view(B, H, W, C)
            t = t.view(B, H, W, C)

            if have_shift:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

                shifted_t = torch.roll(t, shifts=(-shift_size, -shift_size), dims=(1, 2))
                t_windows = window_partition(shifted_t, window_size)  # nW*B, window_size, window_size, C
            else:
                # partition windows
                shifted_x = x
                x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
                shifted_t = t
                t_windows = window_partition(shifted_t, window_size)  # nW*B, window_size, window_size, C

            x = x_windows.view(-1, window_size * window_size, C)
            t = t_windows.view(-1, window_size * window_size, C)

            relative_position = self.en_relative_position[i//2]()
            x = blk(x,relative_position,have_shift,H,W,window_size,shift_size)
            t = blk(t,relative_position,have_shift,H,W,window_size,shift_size)
            
            if (i+self.tau_shift_value)%self.interval_tau == self.interval_tau-1:

                block_group_index = i//self.interval_tau
                with torch.cuda.amp.autocast(enabled=False):
                    x_MoA,t_MoA,x_prompt,t_prompt,aux_MoA=self.blocks_MoA[block_group_index](x,t,task_index) 
                    aux_loss = aux_loss + aux_MoA
                x_prompt_list.append(x_prompt)
                t_prompt_list.append(t_prompt)

                fusion_xt = x_MoA+t_MoA
                fusion_xt = self.FusionLayer[block_group_index] (fusion_xt,H,W)
                
                x = fusion_xt*self.Alpha_encoder[block_group_index] + (1-self.Alpha_encoder[block_group_index])*x
                t = fusion_xt*self.Alpha_encoder[block_group_index] + (1-self.Alpha_encoder[block_group_index])*t


        x = self.norm(x)
        t = self.norm(t)

        encoder_info = {'pH':H,'pW':W,'aux_loss':aux_loss}

        return x,t,x_prompt_list,t_prompt_list,encoder_info


    def forward_decoder(self, x,t, task_index,encoder_info):
        # embed tokens
        H,W,aux_loss = encoder_info['pH'],encoder_info['pW'],encoder_info['aux_loss']
        window_size=14
        shift_size=7
        
        x = self.decoder_embed(x)
        t = self.decoder_embed(t)
        B,L,C = x.shape
        
        pos_embed = self.decoder_pos_embed[:,1:,:].view(1,14,14,-1).permute(0,3,1,2)
        pos_embed=F.interpolate(pos_embed,size=(W,H),mode='bilinear')
        pos_embed=pos_embed.view(1,-1,W*H).permute(0,2,1)

        # add pos embed
        x = x + pos_embed
        t = t+ pos_embed

        x_prompt_list=[]
        t_prompt_list=[]
        # apply Transformer blocks
        for i,blk in enumerate(self.decoder_blocks):
            have_shift = False
            if i % 2 ==1:
                have_shift = True

            x = x.view(B, H, W, C)
            t = t.view(B, H, W, C)

            if have_shift:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

                shifted_t = torch.roll(t, shifts=(-shift_size, -shift_size), dims=(1, 2))
                # partition windows
                t_windows = window_partition(shifted_t, window_size)  # nW*B, window_size, window_size, C
            else:
                shifted_x = x
                # partition windows
                x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

                shifted_t = t
                # partition windows
                t_windows = window_partition(shifted_t, window_size)  # nW*B, window_size, window_size, C

            x = x_windows.view(-1, window_size * window_size, C)
            t = t_windows.view(-1, window_size * window_size, C)


            relative_position = self.de_relative_position[i//2]()
            x = blk(x,relative_position,have_shift,H,W,window_size,shift_size)
            t = blk(t,relative_position,have_shift,H,W,window_size,shift_size)

            if (i+self.tau_shift_value)%self.interval_tau == self.interval_tau-1:
                block_group_index = i//self.interval_tau
                with torch.cuda.amp.autocast(enabled=False):
                    x_MoA, t_MoA,x_prompt,t_prompt,aux_MoA=self.decoder_blocks_MoA[block_group_index](x,t,task_index)
                    aux_loss = aux_loss + aux_MoA
                  
                x_prompt_list.append(x_prompt)
                t_prompt_list.append(t_prompt)

                fusion_xt = x_MoA+t_MoA
                fusion_xt = self.de_FusionLayer[block_group_index] (fusion_xt,H, W)
            
                x = fusion_xt*self.Alpha_decoder[block_group_index] + (1-self.Alpha_decoder[block_group_index])*x
                t = fusion_xt*self.Alpha_decoder[block_group_index] + (1-self.Alpha_decoder[block_group_index])*t

        
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)  #B L C 

        x = self.unpatchify(x,H,W)  #(B, 3, pH, pW)


        return x, x_prompt_list,t_prompt_list,aux_loss

    def getPromptLoss(self,prompt):
        prompt_rgb,prompt_t,de_prompt_rgb,de_prompt_t = prompt

        prompt_rgb = torch.stack(prompt_rgb,dim=0)
        prompt_t = torch.stack(prompt_t,dim=0)
        de_prompt_rgb = torch.stack(de_prompt_rgb,dim=0)
        de_prompt_t = torch.stack(de_prompt_t,dim=0)

        prompt_gt = torch.ones_like(prompt_t) # B 1 H W
        de_prompt_gt = torch.ones_like(de_prompt_t)

        prompt_loss =0.75* self.mse(prompt_rgb+prompt_t,prompt_gt) + 0.25* self.mse(de_prompt_rgb+de_prompt_t,de_prompt_gt)
        return prompt_loss,(prompt_rgb,prompt_t,de_prompt_rgb,de_prompt_t)

    def forward_loss_taskRGBT(self, img_rgb,img_t, pred, prompt,epoch,aux_loss):
        #pred_img = self.unpatchify(pred)
        B,C,H,W = img_rgb.shape
        if self.upsample:
            pred_img =   F.interpolate(pred, size=(H,W))

        pred_img= recover_Norm(pred_img )/255
        img_rgb = recover_Norm(img_rgb)/255
        img_t= recover_Norm(img_t)/255
    
        #prompt B,N,C
        prompt_loss,prompt_tensor =self.getPromptLoss(prompt)

        ssim_loss_rgb = 1 - self.msssim_loss(pred_img, img_rgb, normalize=True)
        ssim_loss_t = 1 - self.msssim_loss(pred_img, img_t, normalize=True)
        ssim_loss=(ssim_loss_rgb+ssim_loss_t)/2
        
        if epoch<self.warmup_epochs:
            pixel_loss =  self.PixelLoss(pred_img,img_rgb,img_t)
            loss = pixel_loss+ 20*prompt_loss +ssim_loss +aux_loss
            loss = {
                "loss":loss,
                "aux_loss":aux_loss,
                "pixel_loss":pixel_loss,
                "prompt_loss":prompt_loss,
                }
            #print(loss)
        else:
            max_grad_loss = self.MaxGradLoss(pred_img,img_rgb,img_t) 
            max_pixel_loss = self.MaxPixelLoss(pred_img,img_rgb,img_t)
            loss = max_grad_loss+ max_pixel_loss + 20*prompt_loss +0.5*ssim_loss +aux_loss
            loss = {"loss":loss,
                "aux_loss":aux_loss,
                "ssim_loss":ssim_loss,
                "max_grad_loss":max_grad_loss,
                "max_pixel_loss":max_pixel_loss,
                "prompt_loss":prompt_loss,
                }


        return loss,pred_img,prompt_tensor
    
    # MSE+SSIM
    def forward_loss_taskMEF(self, img_rgb,img_t, pred, prompt,epoch,aux_loss):

        B,C,H,W = img_rgb.shape
        if self.upsample:
            pred_img = F.interpolate(pred, size=(H,W))

        pred_img= recover_Norm(pred_img )/255
        img_rgb = recover_Norm(img_rgb)/255
        img_t= recover_Norm(img_t)/255
      
        prompt_loss,prompt_tensor =self.getPromptLoss(prompt)

        if epoch<self.warmup_epochs:
            ssim_loss_rgb = 1 -self.msssim_loss(pred_img, img_rgb, normalize=True)
            ssim_loss_t = 1 - self.msssim_loss(pred_img, img_t, normalize=True)
            ssim_loss=(ssim_loss_rgb+ssim_loss_t)/2
            pixel_loss =   self.PixelLoss(pred_img,img_rgb,img_t)
            
            loss =   pixel_loss + 20*prompt_loss +ssim_loss +aux_loss
            loss = {"loss":loss,
                "ssim_loss":ssim_loss,
                "pixel_loss":pixel_loss,
                "prompt_loss":prompt_loss,
                "aux_loss":aux_loss,
                }
        else:
            pixel_loss =   self.PixelLoss(pred_img,img_rgb,img_t)
            pred_img_MEF= pred_img.permute(1,0,2,3).contiguous().view(C,H*B,W)
            img_rgb_MEF= img_rgb.permute(1,0,2,3).contiguous ().view(C,H*B,W)
            img_t_MEF= img_t.permute(1,0,2,3).contiguous ().view(C,H*B,W)
            MEF_SSIM_loss = 1-self.MEFSSIM(pred_img_MEF.unsqueeze(0),torch.stack([img_rgb_MEF,img_t_MEF],dim=0))
            
            max_grad_loss = self.MaxGradLoss(pred_img,img_rgb,img_t)
            loss =  1.2*(max_grad_loss +  20*prompt_loss + MEF_SSIM_loss+aux_loss+pixel_loss)
            loss = {"loss":loss,
                "pixel_loss":pixel_loss,
                "MEF_SSIM_loss":MEF_SSIM_loss,
                "max_grad_loss":max_grad_loss,
                "prompt_loss":prompt_loss,
                "aux_loss":aux_loss,
                }

        return loss,pred_img ,prompt_tensor

    # SSIM + Grad +MaxPixel
    def forward_loss_taskMFF(self, img_rgb,img_t, pred, prompt,epoch,aux_loss):
        B,C,H,W = img_rgb.shape
        if self.upsample:
            pred_img =   F.interpolate(pred, size=(H,W))

        pred_img= recover_Norm(pred_img )/255
        img_rgb = recover_Norm(img_rgb)/255
        img_t= recover_Norm(img_t)/255

        out = self.MFFselect(img_rgb,img_t)
        # max_grad_loss pixel_loss
        max_grad_loss = self.MaxGradLoss(pred_img,out,None)
        pixel_loss = self.MaxPixelLoss(pred_img,out,None)

        #att loss B,N,C
        prompt_loss,prompt_tensor =self.getPromptLoss(prompt)
        ssim_loss_rgb = 1 - self.msssim_loss(pred_img, out, normalize=True)
        ssim_loss=ssim_loss_rgb

        if epoch<self.warmup_epochs:
            loss =  pixel_loss + 20*prompt_loss +ssim_loss +aux_loss
            loss = {"loss":loss,
                "ssim_loss":ssim_loss,
                "pixel_loss":pixel_loss,
                "prompt_loss":prompt_loss,
                "aux_loss":aux_loss,
                }
        else:
            loss =  5*(max_grad_loss + pixel_loss) + 20*prompt_loss +ssim_loss+aux_loss
            loss = {"loss":loss,
                "ssim_loss":ssim_loss,
                "max_grad_loss":max_grad_loss,
                "aux_loss":aux_loss,
                "pixel_loss":pixel_loss,
                "prompt_loss":prompt_loss,
                }
        return loss,pred_img,prompt_tensor

    def forward_loss_split(self, img_rgb,img_t, pred,prompt,task_index,epoch,aux_loss):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        prompt: [# B 1 H W],
        """
        #MSE
        if task_index==0:
             loss,pred_img,prompt_tuple = self.forward_loss_taskRGBT(img_rgb,img_t, pred, prompt,epoch,aux_loss)
        elif task_index==1:
             loss,pred_img,prompt_tuple  = self.forward_loss_taskMEF(img_rgb,img_t, pred, prompt,epoch,aux_loss)
        elif task_index==2:
             loss,pred_img,prompt_tuple  = self.forward_loss_taskMFF(img_rgb,img_t, pred, prompt,epoch,aux_loss)
        return loss,pred_img,prompt_tuple 



    def forward(self, img_rgb,img_t, task_index=0,epoch=0):
        if len(img_rgb.shape) == 5:
            img_rgb = img_rgb.squeeze()
            img_t = img_t.squeeze()
        latent,t,prompt_rgb,prompt_t,encoder_info = self.forward_encoder(img_rgb,img_t, task_index)
        pred,de_prompt_rgb,de_prompt_t,aux_loss = self.forward_decoder(latent,t, task_index,encoder_info)  # [N, L, p*p*3]

        with torch.cuda.amp.autocast(enabled=False):
             loss,pred,prompt_tuple  = self.forward_loss_split( img_rgb,img_t, pred, 
                                        (prompt_rgb,prompt_t,de_prompt_rgb,de_prompt_t),
                                        task_index,epoch,aux_loss)
        return loss, pred,prompt_tuple 



def mae_vit_base_patch16_dec512d8b(config,**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),config=config, **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(config,**kwargs):
    #img_size = (768,1024),
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),config=config, **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(config,**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),config=config, **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks



def maybe_print(s: str, flag: bool):
    if flag:
        print(s)

def load_pretrained_weights(
    model,
    model_ckp_path=None,
    epoch=0,
    model_name=None,
    weights_path=None,
    load_first_conv=True,
    load_fc=True,
    load_repr_layer=True,
    resize_positional_embedding=False,
    resize_len=None,
    embed_dim = None,
    decoder_dim = None,
    verbose=True,
    strict=False,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    print("load_model_info:",model_name,weights_path)

    # Load weights
    state_dict = torch.load(weights_path)
    state_dict = state_dict['model']

    # Modifications to load partial state dict
    expected_missing_keys = []
    for key in expected_missing_keys:
        state_dict.pop(key)

    print("load_model_freeze_state_dict.keys():",state_dict.keys())
    
    freeze_layer = []
    unfreeze = []
    ema_windows_list = []
    ema_MoA_list = []
    if epoch==0:
        ret = model.load_state_dict(state_dict, strict=False)
        for name, parms in model.named_parameters():
            if name in unfreeze:
                parms.requires_grad = True
            elif name in state_dict:
                parms.requires_grad = False
                freeze_layer.append(name)
            else:
                unfreeze.append(name)
            if "relative_position" in name:
                ema_windows_list.append(name)
            if "blocks_MoA" in name:
                ema_MoA_list.append(name)
    else:
        state_dict_ckp = torch.load(model_ckp_path)
        state_dict_ckp = state_dict_ckp['model']
        ret = model.load_state_dict(state_dict_ckp, strict=False)

        for name, parms in model.named_parameters():
            if name in unfreeze:
                parms.requires_grad = True
            elif name in state_dict:
                parms.requires_grad = False
                freeze_layer.append(name)
            else:
                unfreeze.append(name)
            if "relative_position" in name:
                ema_windows_list.append(name)
            if "blocks_MoA" in name:
                ema_MoA_list.append(name)

    maybe_print('Missing keys when loading pretrained weights: {}'.format(
        ret.missing_keys), verbose)
    maybe_print('Unexpected keys when loading pretrained weights: {}'.format(
        ret.unexpected_keys), verbose)

    return freeze_layer,unfreeze,ema_windows_list,ema_MoA_list,model