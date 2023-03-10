# EKD-FWSNet

This repo is the implementation of ["Make Baseline Model Stronger: Embedded Knowledge Distillation in Weight-Sharing Based Ensemble Network"](https://www.bmvc2021-virtualconference.com/assets/papers/0212.pdf). The BMVC paper is extended and submitted to TCSVT.

<table>
    <tr>
    <td><img src="PaperFigs\Fig2.png" width = "100%" alt=""/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt=""/></td>
    </tr>
</table>

## Dataset Preparation

We select CIFAR, tiny-ImageNet, CUB-200 and ImageNet as benchmark datasets. If any questions on finding or constructing experiment datasets, please contact us with the email in the end.

**In the following, we provide the detailed commands for dataset preparation.**

**CIFAR**

     Create CIFAR folder and download CIFAR-10/100 dataset. 
     Move 'cifar-10-batches-py' and 'cifar-100-python' to CIFAR folder.

**tiny-ImageNet**

     Create tiny-imagenet-200 folder and download tiny-imagenet dataset. 
     Move 'tiny_imagenet-200_all' and 'anno' to tiny-imagenet-200 folder.

**CUB-200**
    
     Create CUB2011 folder and download CUB-200 dataset.
     Move 'images', 'labels', 'train.txt' and 'test.txt' to this folder.

**ImageNet**
    
     Create imagenet folder and download ImageNet dataset.
     Move 'train', 'val' to this folder.

## EKD-FWSNet

### Install

1. requirements:
    
    python >= 3.5
        
    pytorch >= 1.4
        
    cuda >= 10.0

### Training

**ImageNet-pretrained backbones** : In some experimental setting, we need to load ImageNet-pretrained backbones. In folder './KDCCB/models/backbone', please download and move 'resnet18-5c106cde.pth', 'resnet34-333f7ec4.pth', 'resnet50-19c8e357.pth' and 'resnet101-5d3b4d8f.pth' to here. No need to move EfficientNet ImageNet-pretrained models here. 

1. Lightweight baseline model optimization:

     ```
     cd ./experiments/Lightweight_ex
     ## under this folder we provide all configuration files (.yaml) of paper-mentioned lightweight EKD-FWSNet. In 'train.sh', you can flexibly switch each configuration files.
     
     sh train.sh
     ```

2. High-efficiency baseline model optimization:

    ```
     cd ./experiments/High-efficiency_ex
     ## under this folder we provide all configuration files (.yaml) of paper-mentioned high-efficiency EKD-FWSNet. In 'train.sh', you can flexibly switch each configuration files.
     
     sh train.sh
     ```

3. Large-scale baseline model optimization:

     ```
     cd ./experiments/Large-scale_ex
     ## under this folder we provide all configuration files (.yaml) of paper-mentioned Large-scale EKD-FWSNet. In 'train.sh', you can flexibly switch each configuration files.
     
     sh train.sh
     ```

### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

open 'train.sh' and add the argument '--is_offline 1'. The best model (.pth) will be automatically stored and loaded.
   
     ```
     cd ./experiments/Lightweight_ex(High-efficiency_ex, Large-scale_ex) 
     
     sh train.sh
     ```

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.
