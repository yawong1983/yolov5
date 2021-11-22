#!/usr/bin/env python
# coding: utf-8

# ### yolov5는 학습 파이프라인이 기존 detection 모델보다 간편한데 비하여, 데이터 전처리 과정이 까다롭다.
# ### 모델 학습 이전에 이미지 리사이징(가로:세로 1:1), bounding box 상대좌표 변환 및 라벨링이 필요하다.
# ### 데이터 전처리 후 train, val 데이터로 분할(80% : 20%)하고, data.yaml, train.txt, val.txt를 생성한다.

# In[1]:


import json, os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[3]:


import matplotlib.pyplot as plt
from PIL import Image


# ### 1번째 데이터셋의 Json을 읽어온다.

# In[4]:


# 데이터 종류에 따른 불러오기 함수
def basic(i):
    code = ['DataSet', 'DataSetII']
    if i == 0:
        f_name = '/data/ubuntu/Data/{0}/{0}.json'.format(code[i])
    if i == 1:
        f_name = '/home/ubuntu/Data/{0}/{0}.json'.format(code[i])

    with open(f_name, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    return json_data    

## 선택된 json에 따라 일반정보 저장
def json_info(i):
    cnt = json_data["totalCountAsset"] #images count
    total_cnt = json_data["totalCountRegions"] # total label count in total images
    total_tags = json_data["totalCountTags"] # 13
    if i == 0:
        path = '/data/ubuntu/Data/DataSet/'
    if i == 1:
        path = '/home/ubuntu/Data/DataSetII/'
    info = [cnt, total_cnt, total_tags, path]
    print("image size :%d, label_image size: %d, tags : %d" %(cnt, total_cnt, total_tags))
    return info

json_data  = basic(0) 
info = json_info(0)


# #### Dataset의 이미지 개수는 5,2929이며, 총 라벨링 개수는 215,846개이며, 13개의 태그로 구성되어있다.

# In[5]:


def tag():
    tags = pd.DataFrame(json_data["tags"])
    tags.index = tags['id']
    return tags
tags = tag()
tags


# ### tags는 id, name, color로 구성되어 있다. 
# ### 태그의 개수는 13개로 7~11번 태그가 교통 장애물이다.

# In[7]:


def df_image():
    assets = json_data["assets"]

    df1 = pd.DataFrame(index=range(0, info[0]), columns=['id', 'name', 'path', 'width', 'height', 'record_time', 'latitude', 'longtitude', 
                                                       'count_labels'])

    for i in range(info[0]):

        images = assets[i]["image"]
        df1.loc[i] = (images['id'], images['name'], images['path'], images["size"]['width'], images["size"]['height'], images['record_time'], 
                        images['latitude'], images['longitude'], images['countRegions'])

    return df1 # image dataframe

df_images = df_image()
df_images


# ### "asset"객체의 정보로 df_images라는 데이터프레임을 생성하였다.
# ### id(식별자), 파일명, 경로, 파일크기, 촬영시간, 라벨수 정보가 있으며, 위/경도 정보는 누락되어 있다.

# In[8]:


def df_label_image():
    assets = json_data["assets"]
    df2 = pd.DataFrame(index=range(0, info[1]), columns=['id', 'sub_id', 'type', 'left', 'top', 'width', 'height', 'tags'])
    sub_sum = 0
    for i in range(info[0]):
        regions = assets[i]["region"] #labelling images info
        for j in range(int(df_images.loc[i, 'count_labels'])):
            df2.loc[sub_sum] = (df_images.loc[i, 'id'], regions[j]['id'], regions[j]['type'], regions[j]['boundingBox']['left'], 
                                           regions[j]['boundingBox']['top'], regions[j]['boundingBox']['width'], 
                                           regions[j]['boundingBox']['height'], regions[j]['tags'][0])
            sub_sum += 1

    return df2
df_label_images = df_label_image()
df_label_images


# ### "region"객체의 정보로 df_label_images라는 데이터프레임을 생성하였다.
# ###  id, sub_id(1+2열이 식별자), 바운딩박스 타입, 바운딩박스 좌표, 태그 타입 정보가 있다.
# ### 바운딩박스 타입은 모두 사각으로 활용성이 없다.

# In[9]:


def label_cnt():
    temp = df_label_images.groupby(['tags']).count().id
    merge_left = pd.merge(temp,tags, how='left', left_on='tags', right_index=True)
    merge_left = merge_left.rename(columns={'id_x':'count'})
    return merge_left[['name', 'count']]
label_cnt = label_cnt()
label_cnt


# ### 각 태그별 바운딩 박스 개수이다. 
# ### 차량, 번호판 개수가 압도적으로 많으며, 8번 스피드 범프, 포트홀 개수는 극히 적다.

# In[10]:


# cutting img check
def img_check(a, o):
    path = info[3]
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
            fpath = os.path.join(path, df_images[df_images.id==i]['path'].iloc[0])
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


# In[34]:


img_check(30, 'o')


# ### 이미지 시현 및 바운딩 박스 확인을 위한 함수를 작성하였다. 
# ### 바운딩박스가 제대로 표시되는 것으로 보아, 좌표값은 신뢰성이 있다.
# ### o:오리지널, r:리사이징 이후 확인

# In[11]:


img_check(38, 'o')


# ### 다만 번호판처럼 일부 태그의 이미지가 식별이 불가한 경우가 많았다. 
# ### 번호판은 블러처리되어 학습에 부적합해 보인다.

# In[142]:


img_check(2303, 'o')


# ### 위 이미지처럼 명도가 낮아 식별이 불가한 태그도 다수 존재하며, 
# ### 명도/채도가 일관성이 없어서 augmentation에 실질적인어려움이 있다. 

# In[ ]:


img_check(46, 'o') #콘
img_check(48, 'o') #스피드범프
img_check(755, 'o') #크랙
img_check(782, 'o') #맨홀
img_check(1924, 'o') #포트홀


# ### 13개의 라벨 중 7~11번 라벨을 학습할 것이다. 
# ### 제공되는 이미지의 포트홀과 크랙은 구분이 어렵다고 판단된다.

# In[19]:


label_cnt[6:11]['count'].sum()#장애물 개수


# ### 바운딩박스 처리된 총 장애물(라벨 7~11)의 개수는 38834개이다.

# In[156]:


df_label_images[(df_label_images.tags >=7) & (df_label_images.tags < 12)].groupby('tags')['width'].mean()
df_label_images[(df_label_images.tags >=7) & (df_label_images.tags < 12)].groupby('tags')['height'].mean()


# ### 7~11번 태크의 평균 크기(가로*세로)이다. 7번(콘)의 너비가 가장 작고, 11번(맨홀)의 높이가 가장 작다.
# ### 또한 이미지간 편차가 크다.

# In[158]:


#라벨 txt작업
#tag 7~11포함 파일 복사
import shutil

e_list = df_label_images[(df_label_images.tags >=7) & (df_label_images.tags < 12)]['id'].unique() #장애물 포함 id


# In[159]:


len(e_list)


# ### 전체 이미지 중 7~11번 라벨의 정보를 e_list 데이터프레임으로 생성하였으며, 총 바운딩박스 개수는 23,459개이다.
# ### 이 데이터셋 전체로 학습을 수행한 결과, 정확도가 낮게 측정되어 일부 데이터를 샘플링하여 다시 학습을 실시하였다.

# In[36]:


def t_imags():
    df7 = df_label_images[df_label_images.tags == 7][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']][:2000] #콘
    df8 = df_label_images[df_label_images.tags == 8][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #스피드범프
    df9 = df_label_images[df_label_images.tags == 9][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #포트홀
    df10 = df_label_images[df_label_images.tags == 10][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']][:2000] #크랙
    df11 = df_label_images[df_label_images.tags == 11][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']][:2000] #맨홀

    t_imags = pd.concat([df7,df8, df9, df10, df11])
    return t_imags

t_imags = t_imags()
t_imags


# ### 콘 2,000개/스피드범프 전체/포트홀 전체/크랙 2,000개/맨홀 2,000개의 정보를 추출하여, t_imags 데이터프레임 생성
# ### 총 라벨수는 6789개이다.

# In[162]:


total_imgs = t_imags['id'].unique()
len(total_imgs)


# ### t_imags의 실제 이미지 개수는 4532개이다.

# In[74]:


# 기 실행된 내용으로 실행하지 말 것
def copy_img(k):
    code = ['', 'secon_']
    path = info[3]
    code2 = code[k]
    for i in total_imgs:
        origin_path = os.path.join(info[3], df_images[df_images.id==i]['path'].iloc[0])
        copy_path = os.path.join("/data/ubuntu/submission/src/yolov5/dataset/images/", code2 + df_images[df_images.id==i]['name'].iloc[0])
        shutil.copyfile(origin_path, copy_path)
copy_img(0)        


# ### 원본 이미지를 데이터셋의 이미지 폴더로 복사하였다.

# In[72]:


# 기 실행된 내용으로 실행하지 말 것
def yolobbox2bbox(k):
    code = ['', 'secon_']
    code2 = code[k]
    for i in total_imgs:
        copy_path = os.path.join("/data/ubuntu/submission/src/yolov5/dataset/labels", code2 + df_images[df_images.id==i]['name'].iloc[0][:-4]+".txt")
        with open(copy_path, 'w') as f:
            for j in t_imags[t_imags.id==i]['sub_id']:
                imgheight,imgwidth = (416, 416)
                x1 = t_imags[(t_imags.id == i)&(t_imags.sub_id==j)]['left'].astype(float)/(1920/416)
                y1 = t_imags[(t_imags.id == i)&(t_imags.sub_id==j)]['top'].astype(float)/(1080/416)
                x2 = t_imags[(t_imags.id == i)&(t_imags.sub_id==j)]['width'].astype(float)/(1920/416)
                y2  = t_imags[(t_imags.id == i)&(t_imags.sub_id==j)]['height'].astype(float)/(1080/416)
                label_l  = t_imags[(t_imags.id == i)&(t_imags.sub_id==j)]['tags'].astype(int)
                print(x1, y1, x2, y2)
                x,y,w,h = (x1.iloc[0], y1.iloc[0], x2.iloc[0], y2.iloc[0])
                x_min = x 
                x_max = (x+w)
                y_min = y
                y_max = (y+h)
                x_center = (x_min+x_max)/2.
                y_center = (y_min+y_max)/2.
                yolo_x = x_center/imgheight
                yolo_w = w/imgwidth
                yolo_y = y_center/imgwidth
                yolo_h = h/imgheight
                label = int(label_l)-7
                temp = str(label)+" "+ str(yolo_x) + " " + str(yolo_y) + " " + str(yolo_w) + " " + str(yolo_h)
                f.write(temp + '\n')

yolobbox2bbox(0)


# ### 라벨링을 하기 위해 t_imags에 해당하는 이미지의 라벨, 좌표값으로 이미지별 txt파일을 생성하였다.
# ### 라벨값에서 -7을 하여 라벨을 0부터 재설정하였다.
# ### 0: cone / 1: speed_bunp / 2 : porthole / 3: crack / 4: manhole (총 5개의 라벨)

# In[28]:


from glob import glob

c_path = "/data/ubuntu/submission/src/yolov5/dataset/"

def c_img_list():

    img_list = glob(c_path + './images/*.jpg')
    label_list = glob(c_path + './labels/*.txt')
    print('이미지:%d개, 라벨:%d개' %(len(img_list), len(label_list)))
    return img_list
img_list = c_img_list()


# ### 이미지와 라벨의 개수는 4532개로 동일하게 생성되었다.

# In[ ]:


# 기 실행된 내용으로 실행하지 말 것
for i in img_list:
    img = Image.open(i)

    img_resize = img.resize((416, 416), Image.LANCZOS)
    img_resize.save(i)


# ### yolov5 학습을 위해서는 가급적 너비와 높이를 동일하게 해야 한다.
# ### 416으로 이미지 크기를 축소하였다.

# In[180]:


img_check(2303, 'r')


# ### 416으로 축소한 이미지와 좌표값이 정상적으로 시현됨을 확인한다. 

# In[27]:


from glob import glob
#포트홀 이미지 추가(3580*2760)
pt_list = glob('./dataset/pothole/images/*.JPG')

print('이미지:%d개' %(len(pt_list)))


# ### 포트홀의 개수가 195개로 적으므로, 데이터를 추가한다.
# ### 1119개의 포트홀 이미지를 데이터셋에 추가하여 학습할 것이다.
# ### 위와 동일한 방식으로 사전에 이미지를 416으로, 좌표값을 yolov5에 적합하게 변경하였다.

# In[210]:


c_img_list()


# ### 포트홀 이미지를 추가한 후의 이미지와 라벨 개수는 5,651개이다.
# ### 이 데이터셋을 활용하여 학습을 수행할 것이다.

# In[211]:


from sklearn.model_selection import train_test_split

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

print(len(train_img_list), len(val_img_list))


# In[213]:


with open(c_path + './train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open(c_path + './val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')


# ### 전체 데이터셋을 학습 데이터셋과 검증 데이터셋으로 구분하고, txt로 저장한다.

# In[20]:


get_ipython().run_line_magic('cd', '/data/ubuntu/pey10/dataset/')
get_ipython().system('git clone https://github.com/ultralytics/yolov5.git')
get_ipython().run_line_magic('cd', '/data/ubuntu/pey10/dataset/yolov5/')
get_ipython().system('pip install -r requirements.txt')


# ### yolov5를 설치한다.

# In[285]:


get_ipython().system('pip install pyyaml==5.4.1')


# In[18]:


import yaml

with open(c_path + './data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(data)

data['train'] = '/data/ubuntu/submission/src/yolov5/dataset/train.txt'
data['val'] = '/data/ubuntu/submission/src/yolov5/dataset/val.txt'
data['names'] = ['cone', 'speed_bump', 'pothole', 'crack', 'manhole']
data['nc'] = 5

with open(c_path + './data.yaml', 'w') as f:
    yaml.dump(data, f)
print(data)


# ### data.yaml을 수정한다.
# ### nc(라벨수):5, train, val txt 경로를 수정한다.

# # 두번째 데이터셋을 추가시키기 위해 위의 과정을 반복한다.

# In[120]:


## 두번째 데이터셋을 학습데이터로 추가한다.
json_data = basic(1) 
info = json_info(1)


# In[111]:


tags = tag()
tags


# In[121]:


df_images = df_image()
df_images


# In[122]:


df_label_images = df_label_image()
df_label_images


# In[123]:


label_cnt = label_cnt()
label_cnt


# In[130]:


label_cnt[6:11]['count'].sum()


# In[132]:


df7 = df_label_images[df_label_images.tags == 7][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #콘
df8 = df_label_images[df_label_images.tags == 8][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #스피드범프
df9 = df_label_images[df_label_images.tags == 9][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #포트홀
df10 = df_label_images[df_label_images.tags == 10][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #크랙
df11 = df_label_images[df_label_images.tags == 11][['id', 'sub_id', 'left','top', 'width', 'height', 'tags']] #맨홀

t_imags = pd.concat([df7,df8, df9, df10, df11])
t_imags


# In[133]:


total_imgs = t_imags['id'].unique()
len(total_imgs)


# In[159]:


copy_img(1)        


# In[164]:


yolobbox2bbox(1)


# In[14]:


img_list = c_img_list()


# In[5]:


### 기 실행된 결과로 실행하지 말 것(파일 리사이즈)
for i in img_list:
    img = Image.open(i)

    img_resize = img.resize((416, 416), Image.LANCZOS)
    img_resize.save(i)


# In[15]:


from sklearn.model_selection import train_test_split

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

print(len(train_img_list), len(val_img_list))


# In[16]:


with open(c_path + './train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open(c_path + './val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')


# In[ ]:




