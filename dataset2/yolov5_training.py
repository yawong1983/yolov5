#!/usr/bin/env python
# coding: utf-8

# In[30]:


import json, os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import import_ipynb


# In[ ]:


get_ipython().run_line_magic('cd', '/data/ubuntu/submission/src/yolov5/yolov5')
get_ipython().system('python train.py --img 416 --batch 10 --epochs 50 --data /data/ubuntu/submission/src/yolov5/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name test2_yolov5s_results')


# ### 앞서 전처리한 데이터셋으로 학습을 수행한다.
# ### 이미지사이즈 : 416, 배치사이즈 : 10, epoches : 50
# ### yolov5s 모델 사용(이전에 medium 활용 시 정확도 저하)
/content/gdrive/MyDrive/deep/yolov5
train: weights=yolov5s.pt, cfg=./models/yolov5s.yaml, data=/content/gdrive/MyDrive/deep/dataset/data.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=50, batch_size=10, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, name=dump_yolov5s_results, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.0-94-g47fac9f torch 1.10.0+cu111 CUDA:0 (Tesla K80, 11441MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=5

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     26970  models.yolo.Detect                      [5, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7033114 parameters, 7033114 gradients

Transferred 342/349 items from yolov5s.pt
Scaled weight_decay = 0.00046875
optimizer: SGD with parameter groups 57 weight, 60 weight (no decay), 60 bias
albumentations: version 1.0.3 required by YOLOv5, but version 0.1.12 is currently installed
train: Scanning '/content/gdrive/MyDrive/deep/dataset/train.cache' images and labels... 4520 found, 0 missing, 0 empty, 3 corrupted: 100% 4520/4520 [00:00<?, ?it/s]
Plotting labels to runs/train/dump_yolov5s_results3/labels.jpg... 

AutoAnchor: 3.05 anchors/target, 0.890 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: WARNING: Extremely small objects found. 710 of 7869 labels are < 3 pixels in size.
AutoAnchor: Running kmeans for 9 anchors on 7867 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.7437: 100% 1000/1000 [00:02<00:00, 370. ㅓ27it/s]
AutoAnchor: thr=0.25: 0.9997 best possible recall, 4.19 anchors past thr
AutoAnchor: n=9, img_size=416, metric_all=0.289/0.744-mean/best, past_thr=0.474-mean: 8,3, 7,20, 19,8, 11,33, 30,14, 26,41, 88,17, 75,60, 223,62
AutoAnchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.
Image sizes 416 train, 416 val
Using 2 dataloader workers
Logging results to runs/train/dump_yolov5s_results3
Starting training for 50 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/49    0.887G    0.1196    0.0113   0.03462        13       416: 100% 452/452 [03:33<00:00,  2.12it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.41it/s]
                 all       1127       1995      0.476      0.121      0.045    0.00866

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/49     1.02G   0.09939   0.01067   0.01476        15       416: 100% 452/452 [03:26<00:00,  2.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.53it/s]
                 all       1127       1995      0.591      0.151      0.184     0.0529

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      2/49     1.02G   0.09488    0.0103   0.01378         9       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.45it/s]
                 all       1127       1995      0.545      0.161      0.116     0.0348

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      3/49     1.02G    0.0943   0.01013   0.01459        12       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.54it/s]
                 all       1127       1995       0.23      0.224      0.129     0.0366

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      4/49     1.02G   0.08997   0.00995   0.01221        17       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.50it/s]
                 all       1127       1995      0.382      0.251      0.237     0.0723

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      5/49     1.02G   0.08517   0.01027   0.01056        21       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.57it/s]
                 all       1127       1995      0.391      0.325      0.273     0.0882

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      6/49     1.02G   0.08422   0.01007   0.01066        14       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.51it/s]
                 all       1127       1995      0.474      0.353      0.321      0.114

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      7/49     1.02G   0.08224   0.01013  0.009486        25       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.60it/s]
                 all       1127       1995       0.39      0.353      0.291      0.108

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      8/49     1.02G   0.08148   0.00982   0.01005        11       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.58it/s]
                 all       1127       1995      0.471      0.374      0.371      0.135

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      9/49     1.02G   0.08049  0.009864  0.009207         8       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.56it/s]
                 all       1127       1995      0.375      0.374      0.331      0.111

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     10/49     1.02G   0.07836  0.009712  0.009291         7       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.59it/s]
                 all       1127       1995       0.47      0.396      0.391      0.143

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     11/49     1.02G   0.07683  0.009511  0.008368         9       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.55it/s]
                 all       1127       1995      0.431      0.447       0.42      0.164

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     12/49     1.02G   0.07591  0.009617  0.007934        14       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.55it/s]
                 all       1127       1995      0.451      0.426      0.383      0.135

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     13/49     1.02G   0.07544  0.009662  0.007703        25       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.58it/s]
                 all       1127       1995      0.428      0.416      0.402      0.152

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     14/49     1.02G   0.07561  0.009543  0.007246        15       416: 100% 452/452 [03:23<00:00,  2.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.56it/s]
                 all       1127       1995      0.434      0.428       0.39      0.145

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     15/49     1.02G   0.07499  0.009614  0.006413        16       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.57it/s]
                 all       1127       1995      0.435       0.47      0.407      0.152

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     16/49     1.02G    0.0733  0.009462  0.007392        25       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.50it/s]
                 all       1127       1995      0.473      0.468      0.441      0.171

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     17/49     1.02G   0.07327  0.009538  0.006495        14       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.54it/s]
                 all       1127       1995      0.514      0.473      0.472      0.188

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     18/49     1.02G   0.07174  0.009638  0.006154        13       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.54it/s]
                 all       1127       1995      0.538      0.477      0.484      0.195

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     19/49     1.02G   0.07142  0.009277  0.006677        11       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.52it/s]
                 all       1127       1995      0.562      0.461      0.477      0.191

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     20/49     1.02G   0.07159  0.009451  0.006498        19       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.47it/s]
                 all       1127       1995      0.502      0.502      0.493      0.196

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     21/49     1.02G   0.07018  0.009336  0.006182        12       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.50it/s]
                 all       1127       1995      0.523      0.527      0.504      0.202

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     22/49     1.02G   0.06987   0.00928  0.005905        14       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.51it/s]
                 all       1127       1995      0.513      0.507      0.483      0.208

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     23/49     1.02G   0.06956   0.00931   0.00615         9       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.50it/s]
                 all       1127       1995      0.545      0.493      0.497      0.217

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     24/49     1.02G   0.06995  0.009425  0.005838        15       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.53it/s]
                 all       1127       1995      0.515      0.564       0.52       0.21

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     25/49     1.02G   0.06883  0.009209  0.005862        13       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.53it/s]
                 all       1127       1995      0.499       0.53      0.504      0.218

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     26/49     1.02G   0.06801   0.00893  0.005704        20       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.54it/s]
                 all       1127       1995      0.535      0.518      0.503      0.212

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     27/49     1.02G   0.06746  0.009018  0.004959        14       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.55it/s]
                 all       1127       1995      0.558      0.538      0.512      0.217

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     28/49     1.02G   0.06666  0.009155   0.00503        14       416: 100% 452/452 [03:23<00:00,  2.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.57it/s]
                 all       1127       1995      0.571      0.529      0.533      0.216

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     29/49     1.02G    0.0671  0.009018   0.00481         9       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.52it/s]
                 all       1127       1995      0.633      0.523      0.542      0.238

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     30/49     1.02G   0.06582  0.008891  0.004927        12       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.55it/s]
                 all       1127       1995      0.579      0.528      0.533      0.243

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     31/49     1.02G   0.06569  0.008985  0.004633         7       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.52it/s]
                 all       1127       1995      0.567      0.556      0.556      0.242

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     32/49     1.02G   0.06522   0.00895  0.005008        16       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.57it/s]
                 all       1127       1995      0.582      0.555      0.541      0.242

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     33/49     1.02G   0.06496  0.008801  0.004418        10       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995      0.595      0.564      0.547      0.243

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     34/49     1.02G   0.06426  0.008906  0.004176        15       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:15<00:00,  3.58it/s]
                 all       1127       1995      0.624      0.542      0.556      0.257

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     35/49     1.02G   0.06427  0.008899  0.004136        17       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.55it/s]
                 all       1127       1995      0.598      0.562      0.558      0.259

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     36/49     1.02G   0.06324  0.008851  0.004004        14       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995      0.641      0.552      0.561      0.256

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     37/49     1.02G   0.06214   0.00882  0.003879        21       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.42it/s]
                 all       1127       1995      0.616      0.574      0.567      0.256

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     38/49     1.02G    0.0622  0.008708  0.003739        10       416: 100% 452/452 [03:26<00:00,  2.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.38it/s]
                 all       1127       1995      0.595      0.589       0.58      0.266

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     39/49     1.02G   0.06157  0.008738  0.003602        11       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995       0.62      0.601      0.587      0.275

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     40/49     1.02G   0.06078  0.008529  0.003683        14       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995       0.65       0.59      0.596      0.276

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     41/49     1.02G   0.06053  0.008557   0.00333        10       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995      0.693       0.57      0.604       0.28

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     42/49     1.02G   0.05964  0.008562  0.003341        22       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.48it/s]
                 all       1127       1995       0.61      0.615      0.592      0.274

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     43/49     1.02G   0.05987  0.008597  0.003592         5       416: 100% 452/452 [03:24<00:00,  2.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.48it/s]
                 all       1127       1995      0.655      0.595      0.606      0.281

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     44/49     1.02G   0.05949  0.008448  0.003066        17       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.42it/s]
                 all       1127       1995      0.694      0.582      0.608      0.286

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     45/49     1.02G   0.05897  0.008386  0.003095        10       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:17<00:00,  3.30it/s]
                 all       1127       1995      0.671      0.595        0.6      0.287

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     46/49     1.02G   0.05831  0.008371  0.002997        12       416: 100% 452/452 [03:26<00:00,  2.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.45it/s]
                 all       1127       1995       0.66      0.603      0.613       0.29

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     47/49     1.02G   0.05921  0.008523  0.002795        14       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.39it/s]
                 all       1127       1995      0.652      0.612      0.614       0.29

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     48/49     1.02G   0.05821  0.008518  0.002971        14       416: 100% 452/452 [03:26<00:00,  2.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.43it/s]
                 all       1127       1995      0.662      0.614      0.618      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     49/49     1.02G   0.05824  0.008321  0.003111        15       416: 100% 452/452 [03:25<00:00,  2.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:16<00:00,  3.47it/s]
                 all       1127       1995      0.678      0.599      0.615      0.293

50 epochs completed in 3.088 hours.
Optimizer stripped from runs/train/dump_yolov5s_results3/weights/last.pt, 14.4MB
Optimizer stripped from runs/train/dump_yolov5s_results3/weights/best.pt, 14.4MB

Validating runs/train/dump_yolov5s_results3/weights/best.pt...
Fusing layers... 
Model Summary: 213 layers, 7023610 parameters, 0 gradients
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 57/57 [00:19<00:00,  2.99it/s]
                 all       1127       1995       0.66      0.614      0.618      0.295
                cone       1127        414      0.715      0.713      0.726      0.344
          speed_bump       1127        121       0.82      0.884      0.929      0.496
             pothole       1127        690       0.57      0.283      0.287     0.0947
               crack       1127        391      0.458      0.389       0.35      0.134
             manhole       1127        379       0.74      0.802        0.8      0.408
Results saved to runs/train/dump_yolov5s_results3
# ### 위 조건으로 학습 수행 시 정확도 평균 66%의 학습결과를 얻었으며, 학습데이터셋이 
# ### 가장 적은 스피드 범프의 정확도가 가장 높게 나타났다. 
# ### 또한 크랙의 정확도가 가장 낮게 측정되었다.

# In[43]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /data/ubuntu/submission/src/yolov5/yolov5/runs/')


# In[ ]:


from IPython.display import Image
import os
get_ipython().run_line_magic('cd', '/data/ubuntu/submission/src/yolov5/yolov5')
# val_img_path = "/data/ubuntu/submission/src/yolov5/test/t_8.jpg"  ##테스트 폴더 이미지 활용
val_img_path = "/data/ubuntu/submission/src/yolov5/dataset/images/14150.jpg"  ##검증(val) 배치이미지 활용
get_ipython().system('python detect.py --weights /data/ubuntu/pey10/yolov5/runs/train/dump_yolov5s_results3/weights/best.pt --img 416 --conf 0.5 --source "{val_img_path}"')
Image(os.path.join('/data/ubuntu/submission/src/yolov5/yolov5/inference/output', os.path.basename(val_img_path)))


# In[30]:


### 이미지를 활용하여 예측결과를 확인한다.
def t_img_show(path):
    fpath = path
    img = Image.open(fpath, 'r')
    plt.subplots(1, figsize=(20,20))
    plt.imshow(img)
t_img_show("/data/ubuntu/submission/src/yolov5/yolov5/runs/detect/exp3/13268.jpg")


# ### best.pt weight 를 적용하여 테스트 이미지에서 맨홀이 예측되었다. 

# # !!!!아래 코드를 실행 전 하단의 모듈로드 및 함수를 실행해야 한다.!!!

# In[28]:


img_check(df_images[df_images.name=='13268.jpg'].id.iloc[0], 'o')


# ### 원본 이미지에서도 맨홀 라벨이 있는 것을 확인할 수 있다.
# ### 정상적으로 예측되었다.

# In[35]:


t_img_show("/data/ubuntu/submission/src/yolov5/yolov5/runs/detect/exp4/29492.jpg"


# In[32]:


img_check(df_images[df_images.name=='29492.jpg'].id.iloc[0], 'o')


# In[37]:


t_img_show("/data/ubuntu/submission/src/yolov5/test/t_8.jpg")


# In[39]:


t_img_show("/data/ubuntu/submission/src/yolov5/yolov5/runs/detect/exp5/t_8.jpg")


# ## 위 이미지 로드 함수를 실행하기 전에 아래의 코드를 순차적으로 실행한 후 이미지를 로드한다.

# In[36]:


####이미지 로드 전 아래를 먼저 수행해야 한다.
from glob import glob
t_list = glob('/data/ubuntu/submission/src/yolov5/test/*.jpg')
for i in t_list:
    img = Image.open(i)

    img_resize = img.resize((416, 416), Image.LANCZOS)
    img_resize.save(i)


# In[16]:


with open('/data/ubuntu/Data/DataSet/DataSet.json') as json_file:
    json_data = json.load(json_file)


# In[17]:


cnt = json_data["totalCountAsset"] 
assets = json_data["assets"]

df_images = pd.DataFrame(index=range(0, cnt), columns=['id', 'name', 'path', 'width', 'height', 'record_time', 'latitude', 'longtitude', 
                                                   'count_labels'])

for i in range(cnt):
    
    images = assets[i]["image"]
    df_images.loc[i] = (images['id'], images['name'], images['path'], images["size"]['width'], images["size"]['height'], images['record_time'], 
                    images['latitude'], images['longitude'], images['countRegions'])

df_images # image dataframe


# In[22]:


total_cnt = json_data["totalCountRegions"] 
df_label_images = pd.DataFrame(index=range(0, total_cnt), columns=['id', 'sub_id', 'type', 'left', 'top', 'width', 'height', 'tags'])
sub_sum = 0
for i in range(cnt):
    regions = assets[i]["region"] #labelling images info
    for j in range(int(df_images.loc[i, 'count_labels'])):
        df_label_images.loc[sub_sum] = (df_images.loc[i, 'id'], regions[j]['id'], regions[j]['type'], regions[j]['boundingBox']['left'], 
                                       regions[j]['boundingBox']['top'], regions[j]['boundingBox']['width'], 
                                       regions[j]['boundingBox']['height'], regions[j]['tags'][0])
        sub_sum += 1

df_label_images


# In[25]:


tags = pd.DataFrame(json_data["tags"])
tags.index = tags['id']


# In[26]:


# cutting img check
def img_check(a, o):
    crop_img = []
    bboxes = []
    if o == "o":
        w1, w2 = 1., 1.
    elif o == 'r':
        w1, w2 = 1920/416, 1080/416
    for i in range(len(df_label_images[df_label_images.id==a])):
        x = df_label_images[df_label_images.id==a]['left'].iloc[i]/w1
        y =  df_label_images[df_label_images.id==a]['top'].iloc[i]/w2
        w =  df_label_images[df_label_images.id==a]['width'].iloc[i]/w1
        h =  df_label_images[df_label_images.id==a]['height'].iloc[i]/w2
        bboxes.append((x, y, w, h))
        
    fig, axs = plt.subplots(1, figsize=(30,30))
    for i in range(a, a+1):
        if o == "o":
            fpath = os.path.join("/data/ubuntu/Data/DataSet/", df_images[df_images.id==i]['path'].iloc[0])
        elif o == "r":
            fpath = "/data/ubuntu/submission/src/yolov5/dataset/images/"+df_images[df_images.id==i]['name'].iloc[0]
        img = Image.open(fpath, 'r')   
        axs.imshow(np.asarray(img))
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
            axs.add_patch(rect)
            axs.axis('off') 
        plt.figure(figsize=(10,10))
        for j in range((int(df_images[df_images.id==i]['count_labels'].iloc[0]))):
            left = bboxes[j][0]
            top = bboxes[j][1]
            width = bboxes[j][2]
            height  = bboxes[j][3]
            label_l  = df_label_images[(df_label_images.id == i)&(df_label_images.sub_id==j)]['tags'].astype(int)

            dim = (left, top, left+width, top+height)
            crop_img.append(img.crop(dim))

            plt.subplot(1,df_images[df_images.id==i]['count_labels'].iloc[0],j+1)
            plt.xticks([])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(crop_img[j])
            plt.xlabel(tags.loc[int(label_l)]['name'])

