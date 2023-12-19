# -*- coding: utf-8 -*-
"""
Description：
    RLIS标注工具配套使用代码
    道路标注脚本，从线要素生成标签（可生成 road 和 null 类）

Created on Fri Dec 15 02:21:59 2023

@author: Car_Dingding
"""

import os
import shutil
import numpy as np
from tqdm import tqdm, trange
import cv2
from matplotlib import pyplot as plt

# merge txt file
def getTruelabelfromTXT(flielist):
    flielist=os.listdir(flielist)
    maxlength=0
    labelall=[]
    # read all file and get img name
    for filename in flielist:
        filefulldir=os.path.join(txtdir,filename)
        with open(filefulldir, 'r',encoding='GBK') as file:
            lines = file.readlines()
            nameflag=len(lines)>maxlength
            if nameflag:
                figname = []
                maxlength=len(lines)
                for line in lines:
                    labelall.append(line)
                    figname.append(line.split('.')[0])
            else:
                for line in lines:
                    labelall.append(line)
    # clear useless record
    truelabel=[]
    for fig in figname:
        selected=[record for record in labelall if record.startswith(fig)]
        longestrecord = selected[0]
        recordlength=len(longestrecord)
        for record in selected:
            if len(record) > recordlength:
                longestrecord = record
                recordlength=len(record)
        truelabel.append(longestrecord)
    return truelabel

# transform
def transstrrecordtolist(labelsrecord):
    listrecord=[]
    for record in labelsrecord:
        split=record.strip().split(' ')
        listrecord.append(split)
    return listrecord

# split name and start scripts
def createlineimage(listrecord):
    for record in tqdm(listrecord):
        filename=record[0].split('/')[-1].split('.')[0]
        record[0]=filename
        createsinglelabelimg(record)
    return

# separate different categories in a single photo
def createsinglelabelimg(record):
    roadfile=os.path.join(roaddir,record[0]+roadsuffix+'.'+lablltype)
    nullfile=os.path.join(nulldir,record[0]+nullsuffix+'.'+lablltype)
    roadlist=[]
    nulllist=[]
    for i in range(1,len(record)):
        id=record[i].split(',')[-1] # get type
        if id=='null':
            nulllist.append(record[i][:-5])
        elif id=='road':
            roadlist.append(record[i][:-5])
        else:
            print("Error")
    createlabel(roadfile,roadlist)
    createlabel(nullfile, nulllist)
    return

# get a pointlist and create the picture label
def createlabel(filename,allpoint):
    image = np.zeros(imgsize, dtype=np.uint8)
    if not allpoint:
        cv2.imwrite(filename, image)
        return
    else:
        for pointlist in allpoint:
            points_str=pointlist.split(',')
            points=[(float(points_str[i]),float(points_str[i+1])) for i in range(0,len(points_str),2)]
            points=[setimagebounds(int(x),int(y)) for x, y in points]
            cv2.polylines(image, [np.array(points)], isClosed=False, color=(255, 255, 255), thickness=roadwidth)
        cv2.imwrite(filename, image)
    return

# prevent points out of bounds
def setimagebounds(x, y):
    x=max(0,min(x,imgsize[0]))
    y=max(0,min(y,imgsize[1]))
    return (x,y)

# easily_labeler
def overlay_Label(imagedir,labeldir,resdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    imglist=os.listdir(imagedir)
    lablist=os.listdir(labeldir)

    for img in tqdm(imglist):
        lab=img.split('.')[0]+'_pred_globle_1randmask_1rand3bold_prob_07.png'
        image = cv2.imread(os.path.join(imagedir,img)).astype('int16')
        mask = cv2.imread(os.path.join(labeldir,lab)).astype('int16')
        masktrue=mask[:,:,0]
        image[:, :, 0] += masktrue
        image[:, :, 1]-= masktrue
        image[:, :, 2] -= masktrue
        image[image<0]=0
        image[image>255]=255
        image.astype('uint8')
        cv2.imwrite(os.path.join(resdir,img), image)
    return

# crop for train
def center_crop(img, target_size=(1024, 1024)):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    top_left_x = max(center_x - target_size[0] // 2, 0)
    top_left_y = max(center_y - target_size[1] // 2, 0)
    cropped_img = img[top_left_y:top_left_y + target_size[1], top_left_x:top_left_x + target_size[0]]
    return cropped_img

# shuffer to label worker
def distribute_images(src_folder, dest_folder_base,  group_size=50, start_index=1):
    if not os.path.exists(src_folder):
        raise ValueError(f"Source directory does not exist: {src_folder}")
    os.makedirs(dest_folder_base, exist_ok=True)
    images = [img for img in os.listdir(src_folder) if img.lower().endswith('.jpg')]
    images.sort()
    for i in range(0, len(images), group_size):
        folder_name = f"data_{start_index:03d}"
        dest_folder = os.path.join(dest_folder_base, folder_name)
        os.makedirs(dest_folder, exist_ok=True)
        for img in images[i:i + group_size]:
            shutil.move(os.path.join(src_folder, img), dest_folder)
        start_index += 1
    return

if __name__ == '__main__':
    # create folder
    if not os.path.exists(roaddir):
        os.makedirs(roaddir)
    if not os.path.exists(nulldir):
        os.makedirs(nulldir)
    # start the script
    createlineimage(transstrrecordtolist(getTruelabelfromTXT(txtdir)))