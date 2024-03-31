import torch
import torch.nn as nn
from model.MMOE import MMoE

class ConvFusionLayer(nn.Module):
    """
    """

    def __init__(
        self,
        dim=1024,
        r = 32,
      
    ):
        super().__init__()
        self.dim = dim
        kersize =3
        self.conv1 = nn.Conv2d(dim,dim//4,(1, 1),bias=False,padding=0)
        self.conv2 = nn.Conv2d(dim//4,dim//4,(kersize, kersize),bias=False,padding=1)
        self.conv3 = nn.Conv2d(dim//4,dim,(1, 1),bias=False,padding=0)
        
    def forward(self, x,H,W):

        feature = x
        B,L,N = feature.shape
        feature = feature.permute(0,2,1).view(B,N,H,W)
        #print(feature.shape)
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        x = feature.view(B,N,H*W).permute(0,2,1)
        #x = torch.cat((cls_token, feature), dim=1)
        #x = self.LayerNorm(x)
        return x



class BiMixtureOfAdapters(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        dim=1024,
        r=16,
        task_num=3,
    ):
        super().__init__()
        self.dim = dim
        self.dimReduction = nn.Linear(dim*2, dim//4, bias=False)
        self.MoA = MMoE(dim//4, dim//4, 4,dim//32, noisy_gating=True, k=2,task_num=3)
        
        #print()
        self.modal_shifts = [nn.Parameter(torch.zeros(dim)).cuda()  for i in range(2*task_num)]
        self.MoA_relu = nn.ReLU()
        self.MoA_sigmoid = nn.Sigmoid()
        self.norm1 = nn.LayerNorm(dim*2)
        self.norm2 = nn.LayerNorm(dim//4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.init_scale_shift()

    def init_scale_shift(self):
        for layer in self.modal_shifts:
            nn.init.normal_(layer, std=.02)
        torch.nn.init.xavier_uniform_(self.dimReduction.weight)
       
    def forward(self, x,t,task_index):
        y = torch.cat([x,t],dim=-1)   #B N C
         # Fsq操作：经池化后输出b*c的矩阵
         # B C 1
        #y = torch.squeeze(self.gap(x_att)) #B N C
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        #print("y=x",y.shape)
        B,N,C = x.shape
        y = self.norm1(y)
        y = self.dimReduction(y)
        y = self.norm2(y)
        y = y.view(B*N,C//4)
        y, aux_loss = self.MoA(y, task_index)
        y = y.view(B,N,C//4)
        prompt_x,prompt_t= torch.chunk(y,2,dim=-1)
        #print("prompt_x0",prompt_x)
        prompt_x = self.gap(prompt_x)
        #print("prompt_x1",prompt_x)
        prompt_x = self.MoA_sigmoid(prompt_x)
       #print("prompt_x2",prompt_x)
        prompt_t = self.gap(prompt_t)
        prompt_t = self.MoA_sigmoid(prompt_t)

     
        # prompt_x =0.5 + 0.01*(prompt_x-0.5)
        # prompt_t =0.5 + 0.01*(prompt_t-0.5)
        # Fscale操作：将得到的权重乘以原来的特征图x
        #print("shape",self.dim,x.shape,y.shape)
        out_x= prompt_x  * x
        out_t= prompt_t  * t
        #out= out + self.SSF_beta.repeat(B).view()
        out_x = torch.add(out_x, self.modal_shifts[task_index*2+0], alpha=1)
        out_t = torch.add(out_t, self.modal_shifts[task_index*2+1], alpha=1)
        return out_x,out_t,prompt_x,prompt_t,aux_loss