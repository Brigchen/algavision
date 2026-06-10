# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:20:41 2019

@author: Brig
"""
import os
import shutil
#from skimage import io
from PIL import Image
#import numpy as np

#%%
#def subdir_to_list():
def make_datasets(original_path, target_path, ratio = 2):
    if os.path.exists(original_path):
        print(original_path+'存在')
    else:
        print(original_path+'没有找到，请确认文件夹名称')   
    isExists=os.path.exists(target_path)
    if not isExists:
        os.makedirs(target_path)
        print(target_path+' 创建成功')
    else:
        print(target_path+' 目录已存在')      
        
    f=open(os.path.join(target_path,'classes_names.txt'),'w')
    f.truncate()
    for _, dirs, _ in os.walk(original_path):
        n=-1
        for dn in dirs:
            print(dn)
            n+=1
            nm='%d/'%n
            subpath=os.path.join(original_path,dn)
            print(subpath)
            files = os.listdir(subpath)
            file_num= len(files)
            print(file_num)
            f.write(str(n)+','+dn+'\n')
            tr='train/'
            vl='val/'
            ts='test/'
            target_tr= os.path.join(target_path,tr,nm)
            target_vl= os.path.join(target_path,vl,nm)            
            target_ts= os.path.join(target_path,ts,nm)
            if os.path.exists(target_tr):
                shutil.rmtree(target_tr)
            if os.path.exists(target_ts):
                shutil.rmtree(target_ts)                
            if os.path.exists(target_vl):
                shutil.rmtree(target_vl)   
            os.makedirs(target_tr)
            os.makedirs(target_ts)
            os.makedirs(target_vl)    
                
            file_names=[os.path.join(subpath, file) for file in os.listdir(subpath)]
        
            for i, file in enumerate(file_names):
                if i%(int(ratio)+2) == 0:
                    shutil.copy(file,target_vl)
                elif i%(int(ratio)+2) == 1:
                    shutil.copy(file,target_ts)
                else:
                    shutil.copy(file,target_tr)    
    f.close()
    print('classes_names.txt produced')
#%%
def label_datasets(root):
    for nm in ["train","val", "test"]:
        f=open(root+'%s.txt'%nm,'w')
        f.truncate()
        data_path=root+'%s/'%nm
        for _,subdirs,_ in os.walk(data_path):
            for dr in subdirs:
                path2=data_path+dr+'/'
                print(path2)
                for img in os.listdir(path2):
                    # if imgtype in name:
                    path3=os.path.join(path2,img) 
                    try:
                        img=Image.open(path3)
                        f.write(path3+','+dr+'\n')
                    except:
                        print('open error:', dr)                                         
        f.close()
        
#%%
if __name__ == '__main__':
    
    original_path=r'E:\Pytorch\Library\WP_genus_enhanced/'
    target_path=r'E:\Pytorch\Training/WP_220429/'
    ratio = 2
    make_datasets(original_path, target_path, ratio)
    label_datasets(target_path)
