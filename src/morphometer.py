# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:20:41 2019

@author: Brig
"""
import os
import shutil
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
from tqdm import tqdm
import datetime
# import matplotlib.pyplot as plt

# for 10X flowCAM image, 1 pixel = 0.4893um as measured
def plot(image,path_out = None, title = None):
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    plt.figure(figsize=(15,15), dpi = 100)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    if path_out:
        plt.savefig(path_out + title + "%s.jpg"%now)
    plt.show()

def plot2(image1, image2, path_out = None, title = None):
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    plt.figure(figsize=(30,15), dpi = 100)
    plt.subplot(1,2,1)
    if len(image1.shape) == 3:
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image1)
    plt.axis('off')
    plt.subplot(1,2,2)
    if len(image2.shape) == 3:
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image2)
    plt.axis('off')
    plt.tight_layout()
    if path_out:
        plt.savefig(path_out + title + "%s.jpg"%now)
    plt.show()
#%% fit light field microscope
def findcnts(img):
    if len(img.shape)==3:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else: 
        img_gray = img
    # print(img.shape)
    # blocksize = int((img.shape[0]+img.shape[1])/4)*2+1
    # img_coi = img_gray.copy()
    # img_blur=cv2.GaussianBlur(img_gray, (5, 5),3,3)
    th, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)  
    mean_gray = img_gray.mean()
    print(f'otsu thres = {th}  mean gray = {mean_gray}')
    if mean_gray > th:
        _, img_th0 = cv2.threshold(img_gray, th-th/15, 255, cv2.THRESH_BINARY_INV)
        # 自适应阈值二值化
        # img_th0 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                     cv2.THRESH_BINARY_INV, blocksize, 20)
    else:
        _, img_th0 = cv2.threshold(img_gray, th-th/15, 255, cv2.THRESH_BINARY)
        # 自适应阈值二值化
        # img_th0 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                     cv2.THRESH_BINARY, blocksize, 20)
    # 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    # plot(img_th0)
    # img_th0 = cv2.erode(img_th0, kernel, iterations=1)
    # plot(img_th0)
    # img_th = cv2.erode(img_th0, kernel, iterations=1)
    # 进行开运算（先腐蚀后膨胀）
    # img_th = cv2.morphologyEx(img_th0, cv2.MORPH_OPEN, kernel)#,iteractions=2)
    # plot(img_th)
    # img_th = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)#,iteractions=2)
    # plot(img_th)                                                                                                                                                                                                          
    img_th0 = cv2.morphologyEx(img_th0, cv2.MORPH_OPEN, kernel)#,iteractions=2) 
    # plot2(img_otsu, img_th0)
    img_th0 = cv2.morphologyEx(img_th0, cv2.MORPH_OPEN, kernel)#,iteractions=2) 
    # plot(img_th)
    
    cnts, _ = cv2.findContours(img_th0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_max=[]
    c_max_area =[]
    for cnt in cnts:
        # print('checkpoint 93')
        area = cv2.contourArea(cnt)
        # print('checkpoint 95')
        if(area < 15):# or (area > 0.2*image.shape[0]*image.shape[1]): # remove cnt < 8 pixel
            continue      
        # print(area)
        c_max.append(cnt)
        c_max_area.append(area)
    c_max = sorted(c_max, key=cv2.contourArea, reverse=True)
    c_max_area = sorted(c_max_area, reverse=True)
    # print(c_max)
    c_sorted = []
    if len(c_max) > 1:
        c_sorted.append(c_max[0])
        prev = c_max[0]
        for cnt in c_max[1:]:
            # 计算差值比例
            ratio = cv2.contourArea(cnt) / cv2.contourArea(prev)
            # print(cv2.contourArea(cnt) , cv2.contourArea(prev))
            if ratio > 0.25:
                c_sorted.append(cnt)
                prev = cnt
            else:
                break
    else:
        c_sorted = c_max
        
    # img_coi = img.copy()
    # cv2.drawContours(img_coi, c_sorted, -1,[0,0,255], -1)
    # # cv2.putText(img_coi, "%s"%fl,(4,10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,255), 1)
    # plot2(img,img_coi)
    # # print(c_max_area)
    # print(f'len of cmax: {len(c_max)}, len of csorted: {len(c_sorted)}')

    return c_sorted

#%%
def length(cnts,coef=1):
    leng = np.mean([cv2.minAreaRect(cnt)[1][0] for cnt in cnts])
    # length, width = rect[1]
    return leng/coef

def width(cnts,coef=1):
    wid = np.mean([cv2.minAreaRect(cnt)[1][1] for cnt in cnts])
    return wid/coef

def perimeter(cnts,coef=1):
    peri = np.mean([cv2.arcLength(cnt,False) for cnt in cnts])
    return peri/coef

def area(cnts,coef=1):
    area=np.mean([cv2.contourArea(cnt) for cnt in cnts])
    return area/(coef**2)

def diameter(cnts,coef=1):
    equi_diameter = np.mean([np.sqrt(4*cv2.contourArea(cnt)/np.pi) for cnt in cnts])
    return equi_diameter/coef

def length_ellipse(cnts,coef=1):
# 方向是对象定向的角度, 主轴和短轴长度。
    a = np.mean([cv2.fitEllipse(cnt)[1][0] for cnt in cnts])
    return a/coef

def width_ellipse(cnts,coef=1):
# 方向是对象定向的角度, 主轴和短轴长度。
    b = np.mean([cv2.fitEllipse(cnt)[1][1] for cnt in cnts])
    return b/coef

def area_ellipse(cnts,coef=1):
    area = np.mean([3.14*cv2.fitEllipse(cnt)[1][1]*cv2.fitEllipse(cnt)[1][0]/4 for cnt in cnts])
    return area/(coef**2)

def vol_ellipse(cnts,coef=1):
    vol = np.mean([(4/3)*(3.14159)*(cv2.fitEllipse(cnt)[1][0]/2)*((cv2.fitEllipse(cnt)[1][0]/2)**2) for cnt in cnts])
    return vol/(coef**3)

def total_vol(cnts,coef=1):
    vols = np.sum([(4/3)*(3.14159)*(cv2.fitEllipse(cnt)[1][0]/2)*((cv2.fitEllipse(cnt)[1][0]/2)**2) for cnt in cnts])
    return vols/(coef**3)

def counts(cnts,coef=1):
    num = len(cnts)
    return num
    # print()

def resize(img):
    if img.shape[1] < 100:
        if img.shape[0] > img.shape[1]:
            n = 100 / img.shape[1]
            img_out = cv2.resize(img, (int(img.shape[1]*n), int(img.shape[0]*n)))
        else:
            n = 100 / img.shape[0]
            img_out = cv2.resize(img, (int(img.shape[1]*n), int(img.shape[0]*n)))
        return img_out, n
    else:
        return img, 1

def dist(A, B):
    return np.linalg.norm(A-B)

def NIG(image):
    # image_blur = cv2.GaussianBlur(image, (5,5),3,3)
    # img_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = int(image.shape[0]/50)
    x_r =image.shape[0]-x
    y = int(image.shape[1]/50)
    y_r =image.shape[1]-y
    p1 = img_lab[x:x+3,y:y+3, :]
    p2 = img_lab[x_r-3:x_r,y_r-3:y_r, :]
    p3 = img_lab[x:x+3,y_r-3:y_r, :]
    p4 = img_lab[x_r-3:x_r,y:y+3, :]
    p = np.concatenate([p1,p2,p3,p4],axis = 0)
    L_mean = np.median(p[:,:,0])
    A_mean = np.median(p[:,:,1])
    B_mean = np.median(p[:,:,2])
    LAB_mean = np.array([L_mean, A_mean, B_mean])
    # print(LAB_mean.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img_out[x,y] = dist(img_lab[x,y,:], LAB_mean)
    return img_out

def ws(img):
    # 1.图像二值化
    if len(img.shape)==3:
        gray = NIG(img)#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else: 
        gray = img
        
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 2.噪声去除
    # im_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # 3.确定背景区域
    sure_bg = cv2.dilate(thresh, kernel, iterations=1)
    # 4.寻找前景区域
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    # dist_transform = dist_transform * (gray-gray.min())/(gray.max()-gray.min())
    # ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    N =int((gray.shape[0] + gray.shape[1])/5)
    if N%2 ==0:
        N += 1
    sure_fg = cv2.adaptiveThreshold(np.uint8((dist_transform-dist_transform.min())/(dist_transform.max()-dist_transform.min())*255),
                                             255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,N,-45)
    # # 5.找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6.类别标记
    ret, markers = cv2.connectedComponents(sure_fg)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers + 1
    # 现在让所有的未知区域为0
    # markers = markers2.copy()
    markers[unknown == 255] = 0
    
    # 7.分水岭算法
    img_coi = img.copy()
    ws = cv2.watershed(img, markers)
    ws2 = ws.copy()
    ws2[ws == -1] = 0
    ws2[ws == 1] = 0
    ws2[ws > 1 ] = 255
    # img[ws == -1] = (0, 0, 255)

    #thresholding a color image, here keeping only the red in the image
    # th=255 - ws2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,2))
    th = cv2.erode(ws2.astype(np.uint8), kernel, iterations=1)
    
    #inverting the image so components become 255 seperated by 0 borders.
    # th3=255 - th2
    
    #calling connectedComponentswithStats to get the size of each component
    # _,output,_,_=cv2.connectedComponentsWithStats(th2,connectivity=8)
    
    # _, th3 = cv2.threshold(output.astype(np.uint8),1,255,cv2.THRESH_BINARY)
    
    cnts,_ = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    c_max=[]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # print(area)
        if(area < 8):# or (area > 0.2*image.shape[0]*image.shape[1]): # remove cnt < 8 pixel
            continue      
        c_max.append(cnt)
    c_max = sorted(c_max, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img_coi, c_max, -1, [0,0,255], thickness=-1)
    # print(len(c_max))
    # plot2(gray, thresh)
    # plot2(th, ws2)
    # plot(img_coi)
    return c_max#, img_coi
    # plot2(sure_bg,sure_fg)
    # plot2(unknown, img)
#%%    
def mm(img, mms, coef1):
    img, coef2 = resize(img)
    coef = coef2/coef1
    lst_op = [index for (index,value) in enumerate(mms) if value == 1]
    # meas = [0]*len(lst_op)
    mm_operations = ["length","width","diameter","area","perimeter","length_ellipse", "width_ellipse","area_ellipse", "vol_ellipse", "counts", "total_vol"]
    meas = [0]*len(lst_op)
    try:
        # cnts = ws(img) #, img_coi
        cnts = findcnts(img)
        # print(len(cnts))
        if len(cnts) > 0:
            for i,x in enumerate(lst_op):
                meas[i] = eval('%s(cnts, coef)'%mm_operations[x])
        else:
            for i,x in enumerate(lst_op):
                meas[i] = 0
        # if np.random.randint(0,100) > 95:
        #     plot(img_coi)
        #     print(meas)
    except Exception as e:
        print(e)
    return meas

#%%
if __name__ == '__main__':
    path = r"C:\Users\brigc\Documents\MATLAB\Python\GUI\ALGAVISION\test\ROIs\t3"
    mms = [1,1,0,0,0,1,0,1,1,1,1]
    coef1 = 0.4893
    for fl in os.listdir(path)[:20]:
        if fl.split('.')[-1].lower() in ["jpg", "png", "tif", "bmp"]:
            print(fl)
            img = cv2.imread(os.path.join(path,fl))
            if img is not None:
                meas = mm(img,mms, coef1)
            # meas = np.array(meas)
            # nb_comp, output, sizes, centroids,th2 = ws(img)
            # print(meas)
            
        #%%
# def cnt_mask(img,cnt):
# # 掩模和像素点
# # 在某些情况下，我们可能需要包含该对象的所有点.可以这样做：

#     mask = np.zeros(img.shape,np.uint8)
#     cv2.drawContours(mask,[cnt],0,255,-1)
#     pixelpoints = np.transpose(np.nonzero(mask))
#     #pixelpoints = cv2.findNonZero(mask)
    
#     return pixelpoints

# def cnt_point(cnt):
# #极点
# #目标最上面，最下面，最左边，最右边的点

#     leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
#     rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
#     topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
#     bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

#     return leftmost,rightmost,topmost,bottommost

# def xyratio(cnt):

#     x,y,w,h = cv2.boundingRect(cnt)
#     aspect_ratio = float(w)/h

#     return aspect_ratio

# def cnt_extent(cnt):
# # 轮廓面积和外接矩形面积的比值
#     area = cv2.contourArea(cnt)
#     x,y,w,h = cv2.boundingRect(cnt)
#     rect_area = w*h
#     extent = float(area)/rect_area
    
#     return extent

# def cnt_solidity(cnt):
# # 轮廓面积与凸包面积的比

#     area = cv2.contourArea(cnt)
#     hull = cv2.convexHull(cnt)
#     hull_area = cv2.contourArea(hull)
#     solidity = float(area)/hull_area
    
#     return solidity

# def cnt_approx(cnt):
# #轮廓的近似,近似到另外一种由更少点组成的轮廓形状

#     perimeter = cv2.arcLength(cnt,False)
#     epsilon = 0.02*perimeter
#     approx = cv2.approxPolyDP(cnt,epsilon,True)
# #imgnew1 = cv2.drawContours(original_img.copy(), approx, -1, (0,0,255), 3)

#     return approx

# def vol_ball(cnt):
#     #以长方体计算 公式为 a*b*c
#     dia = diameter(cnt)
#     vol=(4/3)*np.pi * (dia/2)**3
#     # print(vol)
#     return vol

# def vol_rectangle(cnt):
#     #以长方体计算 公式为 a*b*c
#     rect = cv2.fitEllipse(cnt)
#     MA,ma = rect[1]
#     a = max(MA,ma)
#     b = min(MA,ma)
#     vol=a*b*b
#     # print(vol)
#     return vol            
