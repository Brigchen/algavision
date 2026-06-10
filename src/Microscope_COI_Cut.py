# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:11:32 2019
@author: brigc

update:
    2021/06/10: 1. change rotation to rectangle; 2. use new algorithm for binary
"""
# import os
import cv2
import numpy as np
# from math import fabs, sin, radians, cos
import shutil
import matplotlib.pyplot as plt

#%%
def edge_denoise(img_edge,limit_size):
    _, cnts, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img=img_edge.copy()
    for i in range(len(cnts)):
        cnt = cnts[i]
        area = cv2.contourArea(cnt)
        pxs=cnt.shape[0]+cnt.shape[1]
        
        if(pxs <= limit_size and area < 10):
## thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv2.drawContours(img, cnt, -1, (0,0,0), thickness=-1)
 
    return img

#%%
# def field_type(img_gray):
#     # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     img_gray=cv2.GaussianBlur(img_gray, (5, 5), 0)
#     LT = 12
#     img_b = cv2.Canny(img_gray,LT,LT*3, apertureSize=3, L2gradient=False)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     img_ex = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel,iterations=2)
#     cnts, _ = cv2.findContours(img_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     img_mc = cv2.add(img_gray, 
#                       np.zeros(np.shape(img_gray), dtype=np.uint8), 
#                       mask=img_ex)
#     img_mean = np.mean(np.mean(img_gray))
#     mc_mean = np.mean(np.mean(img_mc[img_mc>0]))
#     print("Target_intensity:", mc_mean)
#     print("Total_intensity:", img_mean)
#     if img_mean > mc_mean:
#         print("Light field image")
#         return True
#     else:
#         print("Dark filed image")
#         return False

#%% plt.imshow(img_b)
    # plt.figure(figsize=(20,20))
    # plt.subplot(2,2,1)
    # plt.imshow(img_gray)
    # plt.subplot(2,2,2)
    # plt.imshow(img_adaptive)
    # plt.subplot(2,2,3)
    # plt.imshow(img_ex)
    # plt.subplot(2,2,4)
    # plt.imshow(img_blank)
    # plt.show()
#%%
def COI_Cut(image,img_gray, th_size=5): 
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    # img_blur=cv2.GaussianBlur(img_gray, (5, 5), sigmaX=0)
    th, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)  
    # _, img_th = cv2.threshold(img_gray, 127,255, cv2.THRESH_BINARY)
    _, img_blank = cv2.threshold(img_gray,210,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_blank = cv2.morphologyEx(img_blank, cv2.MORPH_OPEN, kernel,iterations=1) 
    N =int((img_gray.shape[0] + img_gray.shape[1])/15)
    if N%2 ==0:
        N += 1
    # print(N)
    img_adaptive = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,N,9)
    # img_adaptive = cv2.Canny(img_gray,th/10,th/2, apertureSize=3, L2gradient=False)
    # img_adaptive2 = edge_denoise(img_adaptive, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_ex = cv2.morphologyEx(img_adaptive, cv2.MORPH_CLOSE, kernel,iterations=2) 
    img_ex = img_ex * img_blank
    cnts, _ = cv2.findContours(img_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_max = []
    for i in range(len(cnts)):
        cnt = cnts[i]
        rect=cv2.minAreaRect(cnt)
        (w,h) = rect[1]
        area = w * h
        # area = cv2.contourArea(cnt)
        # print(area)
        if(area < 4*(th_size/2*3.48)**2):# or (area > 0.2*image.shape[0]*image.shape[1]):
            continue                   
        c_max.append(cnt)        
    # print(len(c_max))
    image_coi=image.copy()
    img_list=[]
    center_list = {}
    # points = 0
    intensities = []
    intensities_bg =[]
    for i, cnt in enumerate(c_max):
# cnt = c_max[0]
        M = cv2.moments(cnt)  # 计算轮廓的各阶矩,字典形式
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center_list[str(i)] = [center_x, center_y]
        # angle=rect[2]
        x,y,w,h = cv2.boundingRect(cnt)
        
        if img_gray.shape[1]-1 - (x + w) > 15:       # expend the rectangle with 10 pixels
            x_r = x + w + 14
        else:
            x_r = x + w - 1
        if img_gray.shape[0]-1 - (y + h) > 15:
            y_r = y + h + 14
        else:
            y_r = y + h - 1           
        if y > 15:
            y = y - 15
        if x > 15:
            x = x - 15        
        # print(x,y,x_r,y_r)
        cv2.rectangle(image_coi,(x,y),(x_r,y_r),(255,0,0),2)
        cv2.circle(image_coi,(center_x, center_y), 10,(0,255,128),2)
        cv2.circle(image_coi,(x, y),5,(0,0,255),2)
        cv2.circle(image_coi,(x_r, y_r),5,(0,0,255),2)
        cv2.circle(image_coi,(x, y_r),5,(0,0,255),2)
        cv2.circle(image_coi,(x_r, y),5,(0,0,255),2)
        imgOut = image[y:y_r,x:x_r,:]
        img_list.append([imgOut])
        # intensities.append(img_gray[center_y,center_x])
        intensities.append(np.mean(np.mean(img_gray[y:y_r,x:x_r])))
        intensities_bg.append(np.mean([img_gray[y, x], img_gray[y_r,x], img_gray[y, x_r], img_gray[y_r, x_r]]))
        # print(np.mean(np.mean(img_gray[y:y_r,x:x_r])), np.mean([img_gray[y, x], img_gray[y_r,x], img_gray[y, x_r], img_gray[y_r, x_r]]))
    ratio_intensity = np.mean(intensities)  / np.mean(intensities_bg) 
    # plt.imshow(image_coi)
    return  img_list,image_coi, center_list, ratio_intensity     

#%% background intensity, using 4 corners 10x10
def intensity_bg(img_gray):
    y1 = int(img_gray.shape[0]/10)
    x1 = int(img_gray.shape[1]/10)

    y2 = img_gray.shape[0] - y1 -10
    x2 = img_gray.shape[1] - x1 -10
    
    if img_gray.shape[0] > 20 and img_gray.shape[1]>20:
        
        ints1 = np.mean(np.mean(img_gray[y1:y1+10, x1:x1+10]))
        ints2 = np.mean(np.mean(img_gray[y2:y2+10, x1:x1+10]))
        ints3 = np.mean(np.mean(img_gray[y1:y1+10, x2:x2+10]))
        ints4 = np.mean(np.mean(img_gray[y2:y2+10, x2:x2+10]))
        # img_gray_coi = img_gray.copy()
        
        # cv2.rectangle(img_gray_coi,(x1,y1),(x1+10, y1+10),(255,0,0),2)
        # cv2.rectangle(img_gray_coi,(x1,y2),(x1+10, y2+10),(255,0,0),2)
        # cv2.rectangle(img_gray_coi,(x2,y1),(x2+10, y1+10),(255,0,0),2)
        # cv2.rectangle(img_gray_coi,(x2,y2),(x2+10, y2+10),(255,0,0),2)
        # print(ints1,ints2,ints3,ints4)
        return np.mean([ints1,ints2,ints3,ints4])
    else:
        return np.mean(np.mean(img_gray))
#%%
def COI_Save(img_path):
    # print(img_path)
    th_size = 8
    image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgs, image_coi, center_list, ratio = COI_Cut(image,img_gray, th_size)
    # print("intensity_ratio:", ratio)

    if ratio < 1:
        print("Bright field image")
        return imgs, image_coi, center_list
    else:
        print("Dark field image")
        img_gray = cv2.bitwise_not(img_gray)
        imgs, image_coi, center_list, ratio = COI_Cut(image,img_gray, th_size)
        # print(ratio)
        return imgs, image_coi, center_list

#%% 
# img_path='./image/31.jpg'
# image = cv2.imread(img_path)
# plt.imshow(image)
# # # #%%

# # # # np.max(image)
# # # # np.min(image)
# # # # # #%%
# # # # t = cv2.calcHist([image],[0],None,[256],[0,255])
# # # # plt.plot(t)
# imgs, img_coi, lst = COI_Save(img_path)
# # i1, i2, i3, i4 = COI_Save(img_path)
# # # # # plt.plot(image)
# # #%%
# plt.figure(figsize=(50,30))
# plt.subplot(2,2,1)
# plt.imshow(image)
# plt.subplot(2,2,2)
# plt.imshow(img_blank)
# plt.subplot(2,2,3)
# plt.imshow(img_adaptive)
# plt.subplot(2,2,4)
# plt.imshow(img_ex)
# plt.show()
# plt.imshow(image_coi)
# plt.hist(i1.ravel(), 256, [0,256])
# plt.show()
#%%
# import cv2
# import numpy as np

# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray,(5,5),0)
#     detected_edges = cv2.Canny(detected_edges,
#                                 lowThreshold,
#                                 lowThreshold*ratio,
#                                 apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',detected_edges)

# lowThreshold = 15
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3

# img = cv2.imread('./image/95.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('canny demo')

# cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()