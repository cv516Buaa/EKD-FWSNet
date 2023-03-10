# EKD-FWSNet

This repo is the implementation of ["Make Baseline Model Stronger: Embedded Knowledge Distillation in Weight-Sharing Based Ensemble Network"](https://www.bmvc2021-virtualconference.com/assets/papers/0212.pdf). The BMVC paper is extended and submitted to TCSVT.

<table>
    <tr>
    <td><img src="PaperFigs\Fig2.png" width = "100%" alt=""/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt=""/></td>
    </tr>
</table>

## Dataset Preparation

We select CIFAR, tiny-ImageNet, CUB-200 and ImageNet as benchmark datasets. 

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

## ST-DASegNet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd ST-DASegNet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) For SegFormerb5 based ST-DASegNet training, we provide ImageNet-pretrained backbone here.

We select deeplabv3 and Segformerb5 as baselines. Actually, we use deeplabv3+, which is a more advanced version of deeplabv3. After evaluating, we find that deeplabv3+ has little modification compared to deeplabv3 and has little advantage than deeplabv3.

For LoveDA results, we evaluate on test datasets and submit to online server (https://github.com/Junjue-Wang/LoveDA) (https://codalab.lisn.upsaclay.fr/competitions/424). We also provide the evaluation results on validation dataset.

<table>
    <tr>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Urban.jpg" width = "100%" alt="LoveDA UDA Urban"/></td>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Rural.jpg" width = "100%" alt="LoveDA UDA Rural"/></td>
    </tr>
</table>

1. Potsdam IRRG to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2
     ```

2. Vaihingen IRRG to Potsdam IRRG:

    ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2
     ```

3. Potsdam RGB to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_PotsdamRGB2Vaihingen.py 2
     ```
     
4. Vaihingen RGB to Potsdam IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2PotsdamRGB.py 2
     ```

5. LoveDA Rural to Urban

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2U.py 2
     ```

6. LoveDA Urban to Rural

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_U2R.py 2
     ```

### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. Testing commands

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mIoU   
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mFscore 
     ```

2. Testing cases: P2V_IRRG_64.33.pth and V2P_IRRG_59.65.pth : [google drive](https://drive.google.com/drive/folders/1qVTxY0nf4Rm4-ht0fKzIgGeLu4tAMCr-?usp=sharing)

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mFscore 
     ```
     
     ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mFscore 
     ```

The ArXiv version of this paper is release. [Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation](https://arxiv.org/pdf/2301.05526.pdf)

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MMGeneration](https://github.com/open-mmlab/mmgeneration)
* [DAFormer](https://github.com/lhoyer/DAFormer)
