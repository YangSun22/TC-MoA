## Introduce

This is the official code for the paper Task-Customized Mixture of Adapters for General Image Fusion, the link to the paper is https://arxiv.org/abs/2403.12494.

Next Update before 20240310

## Preparation

###### -checkpoint:

from MAE ([GitHub - facebookresearch/mae: PyTorch implementation of MAE https//arxiv.org/abs/2111.06377](https://github.com/facebookresearch/mae))

```
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
```

```
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
```

#### -dataset:

baiduYun and GoogleDrive TBA

## Environment Installation

Python3.7+torchvision-0.10.0+timm==0.3.2+scipy==1.2.1+setuptools==59.5.0

To be update

```
conda create -n tcmoa python=3.7 
conda activate tcmoa
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.21.6
pip install torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple safetensors
pip install setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy==1.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.3.2  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install six -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple

scikit-image-0.17.2
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple kornia==0.6.5
if BUG: use follow:
import collections.abc as container_abcs
```

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 NCCL_P2P_LEVEL=NVL nohup python -m torch.distributed.launch \
    --nproc_per_node 3 --master_port 22222 \
    main_train.py --config_path ./config/base.yaml \
     > test.log 2>&1 & 
```

## Test

```
CUDA_VISIBLE_DEVICES=0 python  main_predict.py --config_path ./config/predict.yaml 
```
