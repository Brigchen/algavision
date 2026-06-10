# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:13:58 2019

@author: brigc
"""


# coding: utf-8

import numpy as np
import random
import cv2
#import glob
import os
#from scipy import misc
import matplotlib.pyplot as plt
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']
#%%

# def blur(img, blursize = 5, coef = 1, p = 1):
#     # dst = cv2.blur(src, (blursize, blursize))
#     if random.random() < p:
#         img = cv2.GaussianBlur(img, (blursize, blursize), 1)
#     return img


#%%
def blur_sharp(img, percent = 0.05, p = 1):
    # kernels = np.ones([5, 5], np.float32)/25
    if random.random() < p:
        paras = [[0, -1, 1*(random.uniform(1-percent, 1 + percent))], 
                 [-1,0,1*(random.uniform(1-percent, 1 + percent))], 
                 [1 , 2 , 1/16*(random.uniform(1-percent, 1 + percent))], 
                 [2 , 1, 1/16*(random.uniform(1-percent, 1 + percent))]]
        #int((side)*8)
        side, corner, coef = paras[random.randint(0,3)]
        print(side, corner, coef)
        kernels = coef * np.array([[corner, side, corner], [side, 5, side], [corner, side, corner]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernels)
    # cv2.imshow("custom_blur_demo", dst)
    # plt.imshow(dst.astype('uint8'))
    return img
# out = custom_blur_demo(img, percent = 1, p = 1)
#%%
# 随机对比度和亮度 (概率)
def bright(img, percent, p = 1):
    
    lower = int(np.median(img) * ( - percent))
    upper = int(np.median(img) * (percent) )
    # print(lower, upper)
    if random.random() < p:
        rows, cols, chunnel = img.shape
        c= random.uniform(0.9,1.1)
        b = random.randint(lower,upper)
        # print(c, b)
        blank = np.zeros([rows, cols, chunnel], img.dtype)  # np.zeros(img1.shape, dtype=uint8)    
        img = cv2.addWeighted(img, c, blank, 1-c, b)   
        
    return img

#%%
def color_change(img, gamma = 1.25, p=1):
    # img = cv2.imread(file_name,cv2.IMREAD_COLOR)
    if random.random() < p:
        img_ocean = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        
        HSV_img = cv2.cvtColor(img_ocean,cv2.COLOR_BGR2HSV)
        im_hsv_new = np.copy(HSV_img)
        c=np.bincount(np.concatenate((HSV_img[:,0:3,0].flatten(),
                                      HSV_img[:,-3:,0].flatten(),HSV_img[-3:,:,0].flatten(),
                                      HSV_img[0:3,:,0].flatten())))
        d=np.bincount(np.concatenate((HSV_img[:,0:3,1].flatten(),
                                      HSV_img[:,-3:,1].flatten(),HSV_img[-3:,:,1].flatten(),
                                      HSV_img[0:3,:,1].flatten())))
        color = random.randint(0,250)
        sat = random.randint(50,150)
        hue_mode = np.argmax(c)
        sat_mode = np.argmax(d)
    
        if sat_mode > 0:
            
            im_hsv_new[:,:,0] = (HSV_img[:,:,0] + color - hue_mode)%360
            im_hsv_new[:,:,0][im_hsv_new[:,:,0]==0]=color
            im_hsv_new[:,:,1] = (HSV_img[:,:,1] * ((sat+1) / (sat_mode+1))).astype(np.int)
            im_hsv_new[:,:,1][im_hsv_new[:,:,1]==0]=sat
                
            im_out = cv2.cvtColor(im_hsv_new, cv2.COLOR_HSV2BGR) 
        
            fI = im_out/255.0
            out = (np.power(fI, gamma)*255).astype(np.uint8)
        else:
            fI = img/255.0
            out = (np.power(img/255.0, gamma)*255).astype(np.uint8)
    else:
        out = img.copy()
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_ocean, cv2.COLOR_BGR2RGB))
    # plt.title('ocean')
    # plt.figure()
    # plt.imshow(O)
    # plt.title('color change')
    
    return out
#%%
def color_transfer(img, img_tmp, p = 1):
    if random.random() < p:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        img_bg = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_binary)
        img_mc = cv2.add(img, 
                          np.zeros(np.shape(img), dtype=np.uint8), 
                          mask=cv2.bitwise_not(img_binary))
        sc_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2LAB)
        sc_mc = cv2.cvtColor(img_mc, cv2.COLOR_BGR2LAB)
        s_bg_mean, s_bg_std = get_mean_and_std(sc_bg)
        s_mc_mean, s_mc_std = get_mean_and_std(sc_mc)
        # template cut and conversion
        tmp_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        _, tmp_binary = cv2.threshold(tmp_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        tmp_bg = cv2.add(img_tmp, np.zeros(np.shape(img_tmp), dtype=np.uint8), mask=tmp_binary)
        tmp_mc = cv2.add(img_tmp, 
                          np.zeros(np.shape(img_tmp), dtype=np.uint8), 
                          mask=cv2.bitwise_not(tmp_binary))
        dc_bg = cv2.cvtColor(tmp_bg, cv2.COLOR_BGR2LAB)
        dc_mc = cv2.cvtColor(tmp_mc, cv2.COLOR_BGR2LAB)
        d_bg_mean, d_bg_std = get_mean_and_std(dc_bg)
        d_mc_mean, d_mc_std = get_mean_and_std(dc_mc)
        
        #transfer background
        img_bg_n = ((sc_bg-s_bg_mean)*(d_bg_std/s_bg_std))+d_bg_mean
        np.putmask(img_bg_n, img_bg_n > 255, 255)
        np.putmask(img_bg_n, img_bg_n < 0, 0)
        
        # transfer main character
        img_mc_n = ((sc_mc-s_mc_mean)*(d_mc_std/s_mc_std))+d_mc_mean
        np.putmask(img_mc_n, img_mc_n > 255, 255)
        np.putmask(img_mc_n, img_mc_n < 0, 0)
        
        # img_n
        img_bg_out = cv2.cvtColor(cv2.convertScaleAbs(img_bg_n), cv2.COLOR_LAB2BGR)
        img_mc_out = cv2.cvtColor(cv2.convertScaleAbs(img_mc_n), cv2.COLOR_LAB2BGR)
        img_out = img_bg_out + img_mc_out
    else:
        img_out = img.copy()
    return img_out
#%%
def contrast(img, lower = 1.0, upper = 1.6, p = 1):  # 亮度就是每个像素所有通道都加上b    

    if random.random() < p:
        rows, cols, chunnel = img.shape
        c= random.uniform(lower, upper)
        # c = 1.5
        b = 1
        blank = np.zeros([rows, cols, chunnel], img.dtype)  # np.zeros(img1.shape, dtype=uint8)    
        img = cv2.addWeighted(img, c, blank, 1-c, b)    
    #    cv2.imshow("con_bri_demo", dst) 
    return img
# 随机变换大小

#%%
def crop(img, percent, p = 1):

    if random.random() < p:
        h_img, w_img, _ = img.shape
        r_w = random.uniform(0,1) * percent
        r_h = random.uniform(0,1) * percent
        # print(r_w, r_h)
        
        crop_xmin = int(random.uniform(0,r_w) * w_img)
        crop_ymin = int(random.uniform(0,r_h) * h_img)
        crop_xmax = int(random.uniform((1-r_w) * w_img, w_img))
        crop_ymax = int(random.uniform((1-r_h) * h_img, h_img))
        out = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
    else:
        out = img.copy()
    return out

#%%
def deform(img, percent, p = 1):
    if random.random() < p:
        rows, cols, _ = img.shape
        length = max(rows, cols)
        bgc1=int(np.median(np.concatenate((img[:,0,0],img[:,cols-1,0],img[0,:,0],img[rows-1,:,0]))))
        bgc2=int(np.median(np.concatenate((img[:,0,1],img[:,cols-1,1],img[0,:,1],img[rows-1,:,1]))))
        bgc3=int(np.median(np.concatenate((img[:,0,2],img[:,cols-1,2],img[0,:,2],img[rows-1,:,2]))))
        r1 = random.uniform(-1,1)
        r2 = random.uniform(-1,0-r1)
        r3 = random.uniform(0, 0 - r1 - r2)
        r4 = 0 - r1 - r2 - r3
        
        # print(r1,r2,r3,r4)
        r1,r2,r3,r4 =random.sample([r1,r2,r3,r4],4)
        r5,r6,r7,r8 =random.sample([r1,r2,r3,r4],4)
        pts1 = np.float32([[cols*(1-percent), rows*(1-percent)], [cols*percent,rows*(1-percent)], [cols*(1-percent),rows*percent], [cols*percent,rows*percent]])
        pts2 = np.float32([[cols*((1-percent) + (1-percent)*r1), rows*((1-percent) + (1-percent)*r5)], 
                           [cols*(percent + (1-percent)*r2), rows*((1-percent) + (1-percent)*r6)], 
                           [cols*((1-percent) + (1-percent)*r3), rows*(percent + (1-percent)*r7)], 
                           [cols*(percent + (1-percent)*r4), rows*(percent + (1-percent)*r8)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_out = cv2.warpPerspective(img, M, (length, length), borderValue = (bgc1,bgc2,bgc3))
    else:
        img_out = img.copy()
    return img_out


#%% 随机水平反转

def hflip(img, p=1):
    if random.random() < p:
        # img = img.astype(np.float32)
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
    return img

#%%
def resize(img, percent,p = 1):
    if random.random() < p:
        rows, cols, _ = img.shape
        x = random.uniform(percent, 1+percent)
        new_rows = int(rows*x)
        new_cols = int(cols*x)
        # print(new_rows, new_cols)
        img = cv2.resize(img, (new_cols, new_rows))
    return img



#%% 反相
def reverse(img):
    img_out = cv2.bitwise_not(img)
    return img_out



#%%
# 随机逆时针旋转 
def rotation(img, max_angle = 45, p = 1):
    if random.random() < p:

        rows, cols, channel = img.shape
        length = max (rows, cols)
        bgc1=int(np.median(np.concatenate((img[:,0,0],img[:,cols-1,0],img[0,:,0],img[rows-1,:,0]))))
        bgc2=int(np.median(np.concatenate((img[:,0,1],img[:,cols-1,1],img[0,:,1],img[rows-1,:,1]))))
        bgc3=int(np.median(np.concatenate((img[:,0,2],img[:,cols-1,2],img[0,:,2],img[rows-1,:,2]))))
        angle = random.randint(-max_angle, max_angle)

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        M[0,2] += (length - cols) / 2
        M[1,2] += (length - rows) /2

        rotated = cv2.warpAffine(img, M, (length, length), borderValue = (bgc1,bgc2,bgc3))
    else:
        rotated = img.copy()
    return rotated
#%% 饱和度

def saturation(img, lower=1.8, upper=1.9, p = 1):
    img = img.astype(np.float32)
    if random.random() < p:
        img = img /255.0
        img_hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        r = random.uniform(lower,upper)
        # print(r)
        img_hls_new = np.copy(img_hls)
        # img_hls_new[:,:,1] = 1.1 * img_hls_new[:,:,1]
        # img_hls_new[:,:,1][img_hls_new[:,:,1]>1]=1
        # img_hls_new[:,:,1][img_hls_new[:,:,1]<0]=0
        img_hls_new[:,:,2] = r * img_hls_new[:,:,2]
        img_hls_new[:,:,2][img_hls_new[:,:,2]>1]=1
        img_hls_new[:,:,2][img_hls_new[:,:,2]<0]=0
        img_out = cv2.cvtColor(img_hls_new, cv2.COLOR_HLS2BGR) * 255
        img_out = img_out.astype(np.uint8)
        # plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    else:
        img_out = img.copy()
    return img_out
#%%
# def sharpen(img, percent = 1, p = 1):
#     if random.random() < p:
#         img_blur = blur(img, 5, 1)
#         img = cv2.addWeighted(img, 2*percent, img_blur, -1*percent, 0)
#     return img
#%% 随机垂直反转

def vflip(img, p=1):
    if random.random() < p:
        # img = img.astype(np.float32)
        h_img, _, _ = img.shape
        img = img[::-1, :, :]
    return img

#%% color style transfer
def get_mean_and_std(img):
	x_mean, x_std = cv2.meanStdDev(img)
	x_mean = np.hstack(np.around(x_mean, 2))
	x_std = np.hstack(np.around(x_std, 2))
	return x_mean, x_std



#%%
 
def dataset_enhance(old_path, new_path, num = 500, desample = False, max_num = 2000):    
    treats = [
        "blur_sharp(img,percent = 0.02, p = 0.25)",
        "bright(img, percent = 0.3, p = 0.25)",
        "color_change(img, gamma = 1.1, p = 0.25)",
        "contrast(img, lower=0.8, upper= 2, p = 0.25)",
        "crop(img, percent = 0.3, p = 0.25)",
        "deform(img, percent = 0.9, p = 0.25)",
        "hflip(img, p = 0.25)",
        "resize(img, percent = 0.5,p = 0.25)",
        "rotation(img, max_angle = 45, p = 0.25)",
        "saturation(img, 0.2, 4, p = 0.25)",
        "vflip(img, p = 0.25)",
        "color_transfer(img,img_tmp1,1)",
        "color_transfer(img,img_tmp2,0.5)"]
    
    # img = "./test/1.png"    

    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        os.makedirs(new_path)
    else:
        os.makedirs(new_path)
        
    # elif len(os.listdir.)
    for dr in os.listdir(old_path):
        if (os.path.isdir(os.path.join(old_path,dr))):
            dir_path=os.path.join(old_path,dr)
            target_dir = dir_path.replace(old_path, new_path)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if dr != "": #== "Cyano_Cylindrospermopsis": 
                for _, _, fls in os.walk (dir_path):    
                    if len(fls) < num:      
                        shutil.copytree(dir_path, target_dir)
                        for i in range(num-len(fls)):
                            fl = fls[random.randint(0, len(fls)-1)]
                            fl_path = os.path.join(dir_path,fl)
                            try:
                                img = cv2.imread(fl_path, cv2.IMREAD_COLOR)#.astype(np.float32)

                                for j in range(11):
                                    # print(j)
                                    img = eval(treats[j])                              
                                
                                target_fl = os.path.join(target_dir,fl.replace('.', '_%02d.'%i))
                                # print(target_fl)
                                cv2.imwrite(target_fl,img)
                                # plt.figure()
                                # plt.imshow(img.astype('uint8'))
                                # plt.title('%d'%i)
                            except Exception as e:
                                print('Convert image error:%s-%s-%s'%(fl,treats[j], e))
                    else:
                        if desample:
                            fls_select = random.sample(fls, max_num)
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir)
                            for fl in fls_select:
                                fl_path = os.path.join(dir_path,fl)
                                shutil.copy(fl_path, target_dir)
                        else:
                            shutil.copytree(dir_path,target_dir)


#%%
# if __name__ == '__main__':
    # print('')
    # dataset_enhance('./test','./tmp', num = 10, desample = True, max_num = 10)

#%%

if __name__ == '__main__':
    img = cv2.imread("./test/test.png", cv2.IMREAD_COLOR)#.astype(np.float32) 
    # tmp1 = cv2.imread("./test/tmp1.png", cv2.IMREAD_COLOR)#.astype(np.float32)
    # tmp2 = cv2.imread("./test/tmp2.jpg", cv2.IMREAD_COLOR)#.astype(np.float32)
    treats = [
        "blur_sharp(img,percent = 0.02, p = 1)",
        "blur_sharp(img,percent = 0.02, p = 1)",
        "bright(img, percent = 0.3, p = 1)",
        "color_change(img, gamma = 1.1, p = 1)",
        "contrast(img, lower=0.8, upper= 2, p = 1)",
        "crop(img, percent = 0.3, p = 1)",
        "deform(img, percent = 0.9, p = 1)",
        "resize(img, percent = 0.5,p = 1)",
        "rotation(img, max_angle = 45, p = 1)",
        "saturation(img, 0.2, 4, p = 1)",
        "vflip(img, p = 1)",
        "hflip(img, p = 1)",
        ]        

    ss = ["平滑","锐化","亮度","颜色","反差","随机剪切","变形","降分辨率","随机旋转", "饱合度","垂直镜像","水平镜像"]
    
    plt.figure(figsize = (12,10), dpi = 300)
    for i, treat in enumerate(treats):
        out = eval(treat)
        # plt.figure()
        plt.subplot(3,4,i+1)
# plt.text(0.5,0.5,str((2,3,i)),fontsize=18,ha='center')
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.title(ss[i])
    plt.tight_layout()
    plt.show()      
    plt.savefig('./test/saved-treatment.png', dpi = 300)
# out = eval(treats[11])
# plt.imshow(out)                          
#%%
# img_file= '003.png'                       
# img_org = cv2.imread(img_file2, cv2.IMREAD_COLOR)#.astype(np.float32)
# img = img_org.copy()
# img_file2 = "002.jpg"
# img_tmp = cv2.imread(img_file, cv2.IMREAD_COLOR)#.astype(np.float32)
# #%%
# # img_out = img_rotation(img)
# img_out = bright(img)
# img_out2 = contrast(img_out2)
# img_out = color_change(img)
# img_out2 = saturation(img_out)
# img_out = deform(img)
# img_out = reverse(img)
# img_out2 = color_transfer(img_out, img_template)
# #%%
# # # cv2.imshow("original", img)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# plt.imshow(img)
