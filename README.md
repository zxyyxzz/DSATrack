# DSATrack

The official PyTorch implementation of our **NeurIPS 2025** paper:

**Dynamic Semantic-Aware Correlation Modeling for UAV Tracking**

## Let's Get Started

- ### Environment

  Our experiments are conducted with Ubuntu 20.04 and CUDA 11.6.

- ### Preparation

  - Clone our repository to your local project directory.

  - Download the pre-trained weights from [MAE](https://github.com/facebookresearch/mae) or [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), and place the files into the `pretrained_models` directory under DSATrack project path. You may want to try different pre-trained weights, so I list the links of pre-trained models integrated in this project.

    | Backbone Type |                   Model File                   |                       Checkpoint Link                        |
    | :-----------: | :--------------------------------------------: | :----------------------------------------------------------: |
    |  'vit_base'   |          'mae_pretrain_vit_base.pth'           | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) |
    |  'vit_large'  |          'mae_pretrain_vit_large.pth'          | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) |
    |  'vit_base'   |      'deit_base_patch16_224-b5f2ef4d.pth'      | [download](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
    |  'vit_base'   | 'deit_base_distilled_patch16_224-df68dfff.pth' | [download](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth) |

  - Download the training datasets ([LaSOT](http://vision.cs.stonybrook.edu/~lasot/download.html), [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit), [GOT-10k](http://got-10k.aitestunion.com/downloads), [COCO2017](https://cocodataset.org/#download)) and testing datasets ([DTB70](https://github.com/flyers/drone-tracking), [UAVDT](https://sites.google.com/view/grli-uavdt/), [UAV123](https://cemse.kaust.edu.sa/ivul/uav123)) to your disk, the organized directory should look like:

    ```
    --LaSOT/
    	|--airplane
    	|...
    	|--zebra
    --TrackingNet/
    	|--TRAIN_0
    	|...
    	|--TEST
    --GOT10k/
    	|--test
    	|--train
    	|--val
    --COCO/
    	|--annotations
    	|--images
    ```

  - Edit the paths in `lib/test/evaluation/local.py` and `lib/train/adim/local.py` to the proper ones.

- ### Installation

  We use conda to manage the environment.

  ```
  conda create --name dsatrack python=3.9
  conda activate dsatrack
  bash install.sh
  ```
  
- ### Training

  - Training DSATrack-ViT-B
    ```
    python tracking/train.py --script dsatrack --config vitb_256 --mode multiple --nproc 8 --use_wandb 0
    ```

  - Training pruned variants of DSATrack

    ```
    python tracking/train.py --script dsatrack_stu --config vitb_d8 --mode multiple --nproc 8 --use_wandb 0

    python tracking/train.py --script dsatrack_stu --config vitb_d7 --mode multiple --nproc 8 --use_wandb 0

    python tracking/train.py --script dsatrack_stu --config vitb_d6 --mode multiple --nproc 8 --use_wandb 0

    python tracking/train.py --script dsatrack_stu --config vitb_d4 --mode multiple --nproc 8 --use_wandb 0
    ```

- ### Evaluation

  - DTB70, UAVDT, VisDrone2018, UAV123
  
    ```
    python tracking/test.py --tracker_name dsatrack --tracker_param vitb_256 --dataset_name dtb70 --threads 32 --num_gpus 8
    python tracking/test.py --tracker_name dsatrack --tracker_param vitb_256 --dataset_name uavdt --threads 32 --num_gpus 8
    python tracking/test.py --tracker_name dsatrack --tracker_param vitb_256 --dataset_name visdrone --threads 32 --num_gpus 8
    python tracking/test.py --tracker_name dsatrack --tracker_param vitb_256 --dataset_name uav123 --threads 32 --num_gpus 8
    python tracking/analysis_results.py
    ```
