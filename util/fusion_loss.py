import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional  as F_trans
from torch.autograd import Variable

from math import exp
import numpy as np


class SobelxyRGB(nn.Module):
    def __init__(self,isSignGrad=True):
        super(SobelxyRGB, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [[-0.2, 0, 0.2],
                  [-1, 0 , 1],
                  [-0.2, 0, 0.2]]
        kernely = [[0.2, 1, 0.2],
                  [0, 0 , 0],
                  [-0.2, -1, -0.2]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx*1
        kernely = kernely*1
        kernelx = kernelx.repeat(1,3,1,1)
        kernely = kernely.repeat(1,3,1,1)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.relu = nn.ReLU()

    def forward(self,x):
        #R,G,B = x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        if self.isSignGrad:
            return sobelx+sobely
        else:
            return torch.abs(sobelx)+torch.abs(sobely)
        


class MaxGradLoss(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0,isSignGrad=True):
        super(MaxGradLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, *args, **kwargs):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """        
        if im_tir!=None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad),torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(~mask, 0)
            
            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad,rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad



def to_gray(img):
        #print(img.shape)
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        #print("l_imgshape",l_img.shape)
        l_img = l_img.unsqueeze(dim=-3)
        #print("l_imgshape",l_img.shape)
        return l_img


class MaxPixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(MaxPixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """
        #print("im_tir",im_tir)
        if im_tir!=None:
            pixel_max = torch.max(im_rgb, im_tir).detach()
            #pixel_mean = (im_rgb + im_tir)/2.0
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,pixel_max)
        else:
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,im_rgb)
        
        return pixel_loss

    def getmaxpixel(self, im_rgb, im_tir,im_fusion):
        pixel_max = torch.max(im_rgb, im_tir)
        return  im_rgb, im_tir,pixel_max

class PixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """
        #print("im_tir",im_tir)
        if im_tir!=None:
            #pixel_max = torch.max(im_rgb, im_tir).detach()
            pixel_mean = (im_rgb + im_tir)/2.0
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,pixel_mean)
        else:
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,im_rgb)
        
        return pixel_loss



import skimage
from skimage import morphology
from skimage.color import rgb2gray
import torchvision.transforms as transforms
import PIL.Image

def gaussian_SSIM(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_SSIM(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2,normalize=True):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class MaxGradTokenSelect(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(MaxGradTokenSelect, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, im_rgb, im_tir):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """        
        im_rgb_gray =to_gray(im_rgb)
        im_tir_gray =to_gray(im_tir)
        rgb_grad = self.sobelconv(im_rgb_gray)
        tir_grad = self.sobelconv(im_tir_gray)

        im_rgb,info = self.patchify(im_rgb)
        im_tir,_ = self.patchify(im_tir)
        rgb_grad_patch,info_grad = self.patchify(rgb_grad)
        tir_grad_patch,_ = self.patchify(tir_grad)

        rgb_grad,_ = torch.max(rgb_grad_patch,-1)
        tir_grad,_ = torch.max(tir_grad_patch,-1)
        # print("rgb_grad_patch ",rgb_grad_patch.shape)
        # print("tir_grad_patch ",tir_grad_patch.shape)

        AB_mask = (rgb_grad >= tir_grad).unsqueeze(dim=-1)
        # print(AB_mask.shape,AB_mask)
        # print(im_tir.shape,im_rgb.shape)
       # out = im_tir.masked_scatter(AB_mask, im_rgb)
        out =torch.where(AB_mask , im_rgb, im_tir)

        out = self.unpatchify(out,info)
        # im_tir = self.unpatchify(rgb_grad_patch,info_grad)
        # im_rgb = self.unpatchify(tir_grad_patch,info_grad)
        # im_tir = self.unpatchify(im_tir,info)
        # im_rgb = self.unpatchify(im_rgb,info)


        return out# ,im_rgb,im_tir
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        #print(imgs.shape)
        assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0
        H,W,C = imgs.shape[2],imgs.shape[3],imgs.shape[1]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0],C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p*p*C ))
        return x,(H,W,C)

    def unpatchify(self, x,shape):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = shape[0]//16
        w = shape[1]//16
        #print("shape",shape,x.shape)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p,shape[2]))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], shape[2], h * p, w * p))
        return imgs