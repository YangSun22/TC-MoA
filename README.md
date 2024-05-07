## Introduce

This is the official code for the paper Task-Customized Mixture of Adapters for General Image Fusion, the link to the paper is https://arxiv.org/abs/2403.12494.


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
baiduYun: https://pan.baidu.com/s/1R2R58PjJuMaS2P4uwlTBqg?pwd=hyqv  Code:【hyqv】
and GoogleDrive TBA

## Environment Installation

Python3.7+torchvision-0.10.0+timm==0.3.2+scipy==1.2.1+setuptools==59.5.0

To be update

```
conda create -n tcmoa python=3.7 
conda activate tcmoa
pip install  numpy==1.21.6
pip install torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl 
pip install  safetensors
pip install setuptools==59.5.0 
pip install scipy==1.2.1 
pip install timm==0.3.2  
pip install tensorboard 
pip install six 
pip install opencv-python 
pip install h5py 
pip install einops 

scikit-image-0.17.2
pip install   scikit-image
pip install   kornia==0.6.5
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

```
@inproceedings{zhu2024taskcustomized,
  title={Task-Customized Mixture of Adapters for General Image Fusion},
  author={Pengfei Zhu and Yang Sun and Bing Cao and Qinghua Hu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
