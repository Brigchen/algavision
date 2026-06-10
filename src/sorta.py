'''
update: 20260111 support for efficientnetv2
update: 20260107 flowcam cut error
update: 20250909 fix path error, and excel name error
update: 20250314 fix unmatch of morphology measurement error
update: 20250304 sorta version autoupdate
update: 20250219 fix output error
update: 20231204

next: cancel button to stop thread
'''
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
# from PyQt5.QtCore import QStringListModel,QAbstractListModel,QModelIndex
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from child import Ui_Viewer, Ui_Makedataset
import os
import time
import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import datetime, time
from efficientnet_pytorch import EfficientNet
import warnings
import sys
import cv2
import numpy as np
import imageio
import shutil
from imagecut import Microscope_Cut, Flowcam_Cut
import pandas as pd
# import time
from torch.autograd import Variable
# import pyqtgraph as pg
from torch.utils.data import Dataset, DataLoader

# import os
import subprocess

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
from morphometer import mm
from styleframe import StyleFrame
import configparser
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import timm
# import os
# import datetime 
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_FORMATS = ["tif","TIF","tiff","TIFF","jpg","JPG","jpeg","JPEG","gif","GIF","png","PNG","bmp","BMP"]
#%% 获取sorta的修改日期
def get_update_date():
    # 获取当前脚本文件的路径
    current_file_path = 'sorta.py'
    
    try:
        # 获取文件的修改时间戳
        modification_timestamp = os.path.getmtime(current_file_path)
    
        # 将时间戳转换为可读的日期和时间格式
        modification_date = time.strftime('%Y-%m-%d', time.localtime(modification_timestamp))
    
        # 显示文件的修改日期
        print(f"文件 {current_file_path} 的修改日期是: {modification_date}")
        return modification_date
    
    except FileNotFoundError:
        print(f"文件 {current_file_path} 未找到，请检查文件路径是否正确。")
        return None
    except NameError:
        # 在某些环境中，如交互式解释器，__file__ 可能未定义
        print("无法获取当前脚本文件的路径，请确保在脚本文件中运行此代码。")
        return None


#%% autoupdate version
def update_version():
    """自动更新版本号（带Git检查）"""
    try:
        # 读取当前版本号
        try:
            with open('version.txt', 'r') as f:
                version = f.read().strip()
                major, minor, patch = map(int, version.split('.'))
        except FileNotFoundError:
            major, minor, patch = 2, 0, 0
        
        # 检查是否是Git仓库
        try:
            # 检查 .git 目录是否存在
            if not os.path.exists('.git'):
                print('不是 Git 仓库，跳过版本更新')
                return f'{major}.{minor}.{patch}'
            
            # 获取Git状态
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            if git_status:
                # 如果有文件修改，更新版本号
                patch += 1
                if patch >= 10:
                    patch = 0
                    minor += 1
                    if minor >= 10:
                        minor = 0
                        major += 1
                new_version = f'{major}.{minor}.{patch}'
                
                # 保存新的版本号
                with open('version.txt', 'w') as f:
                    f.write(new_version)
                print(f'版本号已更新为: {new_version}')
                return new_version
            else:
                print('没有文件修改，版本号未更新。')
                return f'{major}.{minor}.{patch}'
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('Git 命令执行失败，跳过版本更新')
            return f'{major}.{minor}.{patch}'
            
    except Exception as e:
        print(f'更新版本号时出错: {e}')
        return '2.0.0'

# update_version()

#%%
import pynvml
import psutil

def get_gpu_mem():
    
    pynvml.nvmlInit()
    # handle = len(pynvml.nvmlDeviceGetHandleByIndex(0))# 这里的0是GPU id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
    mem_total = (int(meminfo.total/1024/1024/1024)) 
    
    return mem_total

def get_mem():
    mem_total = psutil.virtual_memory().total/1024/1024/1024
    return mem_total


def get_batchsize(bsize1, gpu_mem1):
    if torch.cuda.is_available():
        mem = get_gpu_mem()
        # print("GPU mem = %d"%mem)
    else:
        mem = get_mem()
        # print("CPU mem = %d"%mem)
    bsize = int(bsize1/gpu_mem1 * mem)
    # print()
    return bsize

#%% 获取当前桌面路径
# import os
def GetDesktopPath():
    return os.path.join(os.path.expanduser("~"), 'Desktop')

# path = os.path.join(GetDesktopPath(),'Outputs')
# print(path)

#%%
def is_img(fl):
    if fl.split(".")[-1] in IMG_FORMATS:
        return True
    else:
        return False
#%% remove broken image

import glob
import re

def check_chinese(path):
    # 使用正则表达式匹配中文符号
    chinese_regex = r'[\u4e00-\u9fff\u3000-\u303f\ufe30-\ufe4f\ufe10-\ufe1f\uff00-\uffef]'
    return re.search(chinese_regex, path) is not None

def check_path(path):
    print('---------------Checking if novel character in directories and image files-------------------')
    if os.path.isdir(path):
        if check_chinese(path):
            print('Chinese character found in directory: %s'%path)
            return False
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                print(os.path.join(root, dir_name))
                if check_chinese(os.path.join(root,dir_name)):
                    print('Chinese character found in directory: %s'%os.path.join(root,dir_name))
                    return False
            # 打印当前目录下的所有文件
            for file_name in files:
                # print(os.path.join(root, file_name))
                if file_name.split('.')[-1] in IMG_FORMATS and check_chinese(os.path.join(root,file_name)):
                    print('Chinese character found in image file: %s'%os.path.join(root,dir_name))
                    return False
        return True

def check_img(img_path, remove_tag):
    try:
        # Image.open(imgs_path).load()
        tmp = cv2.imread(img_path, cv2.IMREAD_COLOR)
        Image.fromarray(cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        print('ERROR: %s, %s' % (img_path, e))
        # error_num += 1
        if remove_tag:
            os.remove(img_path)
            print('Image remove: %s' % img_path)
        else:
            print('Image read error: %s' % img_path)
        return True
    else:
        return False
 
 
def check_imgs(path): 
    # 排查图片的路径
    # path = ''
    img_list = glob.glob(path+"/**/*.*", recursive=True)
    # print(img_list)
    # 是否进行删除
    remove_tag = False #是否移除损坏图片
    # 为True说明出错
    error_num = 0
    valid_num = 0
    for item in img_list:
        if is_img(item):
            # Tag = has_chinese_characters(item)
            Tag = check_img(item, remove_tag)
            if Tag:
                error_num += 1
            else:
                valid_num += 1
    print("check integraty of images: %d images valid, and %d errors"%(valid_num, error_num))

#%%
# path = r'E:\WZY\SORTA\Zoo_Original\210718'
# check_imgs(path)

        
#%%
class Figure_Bar(FigureCanvasQTAgg):    
    def __init__(self, width ,height , dpi = 100):
        self.fig = plt.figure(figsize=(width, height), dpi= dpi)#, edgecolor='#0000FF', facecolor='#666666')
        super(Figure_Bar, self).__init__(self.fig)
        FigureCanvasQTAgg.updateGeometry(self)
 
    def showImage(self, x, y, title):
        self.fig.clear()
        self.myAxes = self.fig.add_subplot(111)#, frameon = True)
        plt.subplots_adjust(top=.95, bottom=0.35, right=.95, left=.075, hspace=0, wspace=0)
        
        self.myAxes.bar(x, y, width = 0.9, align = 'center', label = title) 
        if max(y) < 100:
            self.myAxes.set_ylim(0, 100)
        elif max(y) < 200:
            self.myAxes.set_ylim(0, 200)
        elif max(y) < 500:
            self.myAxes.set_ylim(0, 500)
        elif max(y) < 1000:
            self.myAxes.set_ylim(0, 1000)
        elif max(y) < 2000:
            self.myAxes.set_ylim(0, 2000)
        elif max(y) < 5000:
            self.myAxes.set_ylim(0, 5000)
        elif max(y) < 10000:
            self.myAxes.set_ylim(0, 10000)
        elif max(y) < 20000:
            self.myAxes.set_ylim(0, 20000)
        elif max(y) < 50000:
            self.myAxes.set_ylim(0, 50000)
        elif max(y) < 100000:
            self.myAxes.set_ylim(0, 100000)
        elif max(y) < 500000:
            self.myAxes.set_ylim(0, 500000)
        # self.myAxes.set_title(title, fontsize )
        self.myAxes.set_ylabel('Counts')
        self.myAxes.set_xlabel("Taxa")
        # self.myAxes.title.set_size(20)
        self.myAxes.xaxis.label.set_size(12)
        self.myAxes.yaxis.label.set_size(12)
        self.myAxes.tick_params(axis='x', labelsize= 10, rotation = 90)
        self.myAxes.tick_params(axis='y', labelsize= 10)
        plt.legend(loc='upper right', fontsize=15) #, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
            #ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
        # self.fig.tight_layout()
        # self.myAxes.set_aspect(1)
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # self.fig.set_size_inches(8,6)
        # plt.show()

#%%
class idenCNN(QThread):
    """
    对图片进行鉴定和测量
    """
    message_signal = pyqtSignal(str)
    data_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(idenCNN, self).__init__()
        self.modeltype = kwargs.get('modeltype')
        # self.modelname = kwargs.get('modelname')
        self.imgs_path = kwargs.get('imgs_path')
        self.output_path = kwargs.get('output_path')
        self.export_yesno = kwargs.get('export')
        self.export_path = kwargs.get('export_path')
        self.mm_yesno = kwargs.get('mm')
        self.mms = kwargs.get('mms')
        self.mm_select = kwargs.get('mm_select')
        self.pxsize = kwargs.get('pxsize')
        self.device_type = kwargs.get('device_type')
        self.level = kwargs.get('level')
        self.image_type = kwargs.get('image_type')
        # print("image path:%s"%self.imgs_path, "morphology measure:%s"%self.mm_yesno,
        #       "export yesno:%s"%self.export_yesno, "export path%s"%self.export_path, 
        #       "batch size:%d"%self.batch_size)
        # self.signal.connect(self.refresh)
    
    def run(self):
        today = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        parser = argparse.ArgumentParser(description='PyTorch Test')
        parser.add_argument('--path_model', default=r"./Models/%s/"%self.modeltype,  # ##############         CHANGE!!! 
                            type=str)         
   
        parser.add_argument('--model_type', default='efficientnet-b3',  # ##############         CHANGE!!! 
                            type=str)      

        parser.add_argument('--path_imgs', default=self.imgs_path,  # ##############         CHANGE!!! 
                            type=str,help='path of imgs')     
 
        parser.add_argument('--path_out', default=self.output_path,  # ##############         CHANGE!!! 
                            type=str,help='path of results output')             
        parser.add_argument('--path_save', default=self.export_path,  # ##############         CHANGE!!! 
                            type=str, help='path of image save')       
        
        parser.add_argument('--size', default=224, type=int)        
        parser.add_argument('--features', default=100, type=int)    
        
        # parser.add_argument('--drop_out',  default=0.1, type=float)   
        
        parser.add_argument('--batch_size', default=256, type=int)   
        
        parser.add_argument('--mm_yesno', default=self.mm_yesno, type=bool)   

        parser.add_argument('--level', default=self.level, type=int)  
        parser.add_argument('--device_type', default=0, type = int)
        parser.add_argument('--image_type', default= 0, type = int)
        
        args = parser.parse_args()
        # argsDict = args.__dict__    
        try:
            
            f = open(os.path.join(args.path_model, "setting.log"), "r", encoding="utf-8")
            for line in f:
                if "model_type:" in line:
                    args.model_type = str(line.split(":")[1].strip())
                   
                if "imgsize:" in line:
                    args.size = int(line.split(":")[1])
                    
                if "features:" in line:
                    args.features = int(line.split(":")[1])
                    self.features = args.features
                    
                if "bsize:" in line:
                    args.batch_size = int(line.split(":")[1])
                    
                if "device_type:" in line:
                    args.device_type = int(line.split(":")[1])
                    self.device_type = args.device_type
                    
                if "image_type:" in line:
                    # setting.log 的 image_type 只应影响 image_type，而不能覆盖 device_type
                    args.image_type = int(line.split(":")[1])
                    self.image_type = args.image_type
            
            # f = open(os.path.join(args.path_model, "setting.log"), "r", encoding="utf-8")
            # for line in f:
                                    
                if "gpu_mem:" in line:
                    gpu_mem1 =  int(line.split(":")[1])
                    print("get from log file: gpu_mem1 = %d"%gpu_mem1)
                    args.batch_size = get_batchsize(args.batch_size, gpu_mem1)
                    if args.batch_size < 1:
                        args.batch_size = 1
                    print("adjust batchsize to %d:"%args.batch_size)
                                
            # self.label_status.setText("")
            self.message_signal.emit("Processing... \n Setting loaded\n model: %s; image size:%d; feature number: %d"%(args.model_type, args.size, args.features))
            # QApplication.processEvents()
        except Exception as e:
            self.message_signal.emit("Load setting error: %s"%e)
            # QApplication.processEvents()
        
    # try:
        #%% load class names
        # time.sleep(2)
        print("finally, bsize = %d"%args.batch_size)
        path_cn=os.path.join(args.path_model,"classes_names.txt")  ##############CHange!'
        print(f'Class name txt path: {path_cn}')
        cn_df = pd.read_csv(path_cn, header=None, names=['No','Class_Names'])
        classes = list(cn_df.iloc[:,1])
        #%% load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.message_signal.emit(device)    
        if args.model_type.startswith('tf_efficientnetv2'):  
            # EfficientNetV2模型加载  
            # import timm  
            model_cnn = timm.create_model(  
                args.model_type,   
                pretrained=False,   
                num_classes=self.features  
            )  
            model_path = os.path.join(args.path_model, 'checkpoint.pt')  
            model_cnn.load_state_dict(torch.load(model_path, map_location=device)) 
        else:
            model_cnn = EfficientNet.from_pretrained(args.model_type)#, dropout_rate = args.drop_out)
            model_cnn._fc.out_features = args.features
            if torch.cuda.is_available():
                model_cnn.load_state_dict(torch.load(os.path.join(args.path_model , 'checkpoint.pt'), map_location='cuda:0'))
            else:
                model_cnn.load_state_dict(torch.load(os.path.join(args.path_model , 'checkpoint.pt'), map_location=torch.device('cpu')))
        model_cnn.to(device)
        model_cnn.eval()
        self.message_signal.emit("Processing....\n Model loaded, and starting test") 
        if not os.path.exists(args.path_out):
            os.makedirs(args.path_out)
        # dfs= pd.DataFrame()
        path_xls=os.path.join(args.path_out,f'CNN_Classes_Counts_{os.path.basename(args.path_imgs)}_{today}.xlsx')
        print(f'Excel path: {path_xls}')
        check_imgs(args.path_imgs)
        num_total = 0
        num_dr = 0
        num_processed = 0
        # QApplication.processEvents()
        print('391 :', self.level, self.mm_yesno)
        # Robust directory handling:
        # If root folder (level0) has no images, but it contains sub-folders with images,
        # users expect "Sub-directories(level1)" behavior even if UI state mismatched.
        try:
            root_entries = os.listdir(args.path_imgs)
        except Exception:
            root_entries = []
        direct_img_total = sum(
            is_img(x)
            for x in root_entries
            if os.path.isfile(os.path.join(args.path_imgs, x))
        )
        subdirs = [
            d for d in root_entries
            if os.path.isdir(os.path.join(args.path_imgs, d))
        ]
        effective_level = 1 if (self.level == 1 or (direct_img_total < 1 and len(subdirs) > 0)) else 0
        print("effective_level:", effective_level, "(ui level:", self.level, ")")

        if effective_level == 1:
            for dr in subdirs:
                # print(dr)
                num_total += sum([is_img(x) for x in os.listdir(os.path.join(args.path_imgs, dr))])
                num_dr += 1
                # print(num_total)
            if len(subdirs) < 1:
                self.message_signal.emit("No subdirectory in current working folder, please check it！")
                return
            with StyleFrame.ExcelWriter(path_xls) as writer:
                for i, imgdir in enumerate(os.listdir(args.path_imgs)):
                    data_path=os.path.join(args.path_imgs, imgdir)
                    # print(data_path)
                    # QApplication.processEvents()
                    mm_nms = self.mm_select#[mm_list[i] for (i,x) in enumerate(self.mms) if x == 1]
                    if os.path.isdir(data_path):
                        self.message_signal.emit("Processing...\n Current Image Folder: %s"%imgdir)
                        # QApplication.processEvents()
                        tf = transforms.Compose([transforms.Resize((args.size,args.size)),
                                                  transforms.ToTensor(), 
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
                        preds_all = []
                        if self.mm_yesno:
                            meas_all = []
                        vol = 0
                        self.progress_signal.emit([len(os.listdir(data_path))-1, None, None, num_processed])
                        if self.device_type == 2 or self.image_type == 1:
                            try:
                                print('loading images')
                                fls, imgs = ImLoader(data_path)
                                print('file size is %d'%len(fls))
                                test_data=MyDataset(imgs=imgs, transform= tf) #transforms.ToTensor())        
                                test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)#, num_workers = 4) 
                                img_count = len(test_data)
                                # print("img segmented: %d"%img_count)
                                # self.message_signal.emit(img_count)
                                pred_score=np.zeros((img_count))
                                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                # print("Testing")
                                with torch.no_grad():
                                    # model_cnn.eval()
                                    preds = []
                                    meas = []
                                    for j, imgdata in enumerate(test_loader,0):
                                        imgs_tensor, labels = imgdata
                                        # plot(imgs[0])
                                        batch_x = imgs_tensor.to(device)
                                        outputs = model_cnn(batch_x)
                                        pred = torch.max(outputs, 1)[1]       
                                        pred_numpy = pred.cpu().numpy().tolist()
                                        preds += pred_numpy
                                        # print('444: ', pred.cpu(), pred_numpy)
                                        num_processed += args.batch_size
                                        self.progress_signal.emit([None, i-1, (j-1)*args.batch_size, num_processed])
                                        self.message_signal.emit("Processing...\n Current Image Folder: %s \n Image: %d"%(imgdir, j*args.batch_size))                                        
                                        # self.message_signal.emit(pred_names)
                                    for k, pred_name in enumerate(preds):
                                        # print('450', pred_name)
                                        try: 
                                            if self.export_yesno == 1:
                                                if int(pred_name) > -1: # or int(pred_name)== 3:
                                                    new_path = os.path.join(self.export_path, imgdir,classes[int(pred_name)]) 
                                                    if not os.path.exists(new_path):
                                                        os.makedirs(new_path)
                                                    shutil.copy(fls[k], new_path)      
                                            if self.mm_yesno: 
                                                # print('459 %d:%d'%(i,pred_name))
                                                if int(pred_name) > -1:
                                                    self.message_signal.emit("Measuring...\n Current Image Folder: %s \n Image: %d"%(data_path, k)) 
                                                    self.progress_signal.emit([None, 0, k, k])
                                                    mea = mm(imgs[k],self.mms,self.pxsize)
                                                    # print('%d:%d'%(i,pred_name))
                                                    # print('465:', mea)
                                                    meas.append(mea)
                                                else:
                                                    mea = [np.nan]*self.mms.count(1)
                                                    # print(mea)
                                                    meas.append(mea)
                                                    # print('%d:%d'%(i,pred_name))
                                            # self._signal.emit([preds, meas, None])
                                            # print('473:', len(meas))
                                        except Exception as e:
                                            print("Error with image measurement or write:%s"%str(e))
                                            # self._signal.emit([preds, meas, e])
                                    
                                preds_all += preds
                                # print("mark4")
    #                                     # print(len(preds_all))
                                if self.mm_yesno:
                                    meas_all += meas
                                    
                                # print('484 ', len(meas))
                                self.data_signal.emit([preds_all, imgdir])
                                # QApplication.processEvents()
                                # print("mark5")
                            except Exception as e:
                                self.message_signal.emit(str(e))
                                print('490', e)
                                # QApplication.processEvents()
                            
                            
                        else:
                            for j, fl in enumerate(os.listdir(data_path)):
                                # print(fl)
        #                         # time.sleep(1)
                                if "summary.txt" in fl:
                                    f = open(os.path.join(data_path, fl), "r", encoding="utf-8")
                                    # self.message_signal.emit("Processing...\n Current Image Folder: %s \n Image: %s"%(imgdir, fl))
                                    # QApplication.processEvents()                    
                                    for line in f:
                                        if "Fluid Volume Imaged:" in line:
                                            vol = float(line.split(":")[1].replace("ml", ""))
                                            print("Volume imaged = ", vol)
                                        # else:
                                        #     print('No Volume found!')
                                if is_img(os.path.join(data_path,fl)):
                                    num_processed += 1
                                    self.progress_signal.emit([None, i-1, j-1, num_processed])
                                    self.message_signal.emit("Processing...\n Current Image Folder: %s \n Image: %s"%(imgdir, fl))
                                    QApplication.processEvents()                    
                                    
                                    if (fl.split(".")[-1] in IMG_FORMATS) & ('cal_image' not in fl):
                                        # rintrint('mark1')
                                        try:
                                            if self.device_type == 0:
                                                imgs = Flowcam_Cut(os.path.join(data_path, fl))
                                                print(len(imgs))
                                            elif self.device_type == 1:
                                                imgs = Microscope_Cut(os.path.join(data_path, fl))
                                                
                                            print("%d particles found in %s"%(len(imgs), fl))
    
                                            test_data=MyDataset(imgs=imgs, transform= tf) #transforms.ToTensor())        
                                            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)#, num_workers = 4) 
                                            img_count = len(test_data)
                                            # print("img segmented: %d"%img_count)
                                            # self.message_signal.emit(img_count)
                                            pred_score=np.zeros((img_count))
                                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                            # print("Testing")
                                            with torch.no_grad():
                                                # model_cnn.eval()
                                                preds = []
                                                meas = []
                                                for _, imgdata in enumerate(test_loader,0):
                                                    imgs_tensor, labels = imgdata
                                                    # plot(imgs[0])
                                                    batch_x = imgs_tensor.to(device)
                                                    outputs = model_cnn(batch_x)
                                                    pred = torch.max(outputs, 1)[1]
                                                    pred_numpy = pred.cpu().numpy().tolist()
                                                    preds += pred_numpy
                                                    # self.message_signal.emit(pred_names)
                                                # print('547 :', len(preds))
                                                for k, pred_name in enumerate(preds):
                                        
                                                    try: 
                                                        if self.export_yesno == 1:
                                                            if int(pred_name) > -1: # or int(pred_name)== 3:
                                                                new_path = os.path.join(self.export_path, imgdir, classes[int(pred_name)]) 
                                                                if not os.path.exists(new_path):
                                                                    os.makedirs(new_path)
                                                                cv2.imwrite(os.path.join(new_path, fl.split('.')[0]+'_%s_ROI_%06d.png'%(imgdir,k)), imgs[k])       
                                                        if self.mm_yesno: 
                                                            # print('557 %d:%d'%(i,pred_name))
                                                            
                                                            if int(pred_name) > -1:
                                                                mea = mm(imgs[k],self.mms,self.pxsize)
                                                                
                                                                # print(mea)
                                                                meas.append(mea)
                                                            else:
                                                                mea = [np.nan]*self.mms.count(1)
                                                                # print(mea)
                                                                meas.append(mea)
                                                                # print('%d:%d'%(i,pred_name))
                                                        # self._signal.emit([preds, meas, None])
                                                        # print(len(preds),len(meas))
                                                    except Exception as e:
                                                        print("Error with image measurement or write:%s"%str(e))
                                                        # self._signal.emit([preds, meas, e])
                                                
                                            preds_all += preds
                                            if self.mm_yesno:
                                                meas_all += meas
                                            self.data_signal.emit([preds_all, imgdir])
                                            # print("mark5")
                                        except Exception as e:
                                            self.message_signal.emit(str(e))
                                            print('567',e)
    
                        count = []
                        if self.mm_yesno:
                            df_meas = pd.DataFrame(meas_all,columns=mm_nms)                           
                            df_meas.replace([np.inf, -np.inf], np.nan, inplace=True)
                            print("length of morphology measures: %d"%len(df_meas))
                            # print(df_meas)
                            # print('599: ', preds_all)
                            # print(args.features)
                            df_meas['Prediction'] = preds_all
                            df = pd.DataFrame(columns = mm_nms+['Prediction'])
                            # print('594 ', args.features, type(args.features))
                            # print('595 ', len(df_meas), type(df_meas))
                            for n in range(args.features):
                                df_i = df_meas[df_meas["Prediction"] == n].mean()
                                df.loc[n] = df_i             
                                count.append(preds_all.count(n))
                            if vol > 0:
                                df['Counts(ind/ml)'] = [float(x)/vol for x in count]
                                if 'Counts(/ind)' in mm_nms:
                                    df['Counts(cells/ml)'] = df['Counts(ind/ml)'] * df['Counts(/ind)']
                                if 'Total_vol(/ind)' in mm_nms:
                                    df['Total_vol(/ml)'] = df['Counts(ind/ml)'] * df['Total_vol(/ind)']  
                            else:
                                df['Counts(ind)'] = count
                                if 'Counts(/ind)' in mm_nms:
                                    df['Counts(cells)'] = df['Counts(ind)'] * df['Counts(/ind)']
                                if 'Total_vol(/ind)' in mm_nms:
                                    df['Total_vol'] = df['Counts(ind)'] * df['Total_vol(/ind)']  
                                    
                        else:
                            for m in range(args.features):           
                                count.append(preds_all.count(m))
                        
                            if vol > 0:
                                nm = 'Counts(ind/ml)'
                                df = pd.DataFrame([float(x)/vol for x in count], columns=[nm])
                            else:
                                nm = 'Counts(ind)'
                                df = pd.DataFrame(count, columns=[nm])
                        # df.to_csv(os.path.join(args.path_out, "{imgdir}_{today}.csv"))
                            # dfs[imgdir_df]=df
                        # df['Class_Names']=cn_df['Class_Names']
                        df.insert(0, 'Class_Names', cn_df['Class_Names'])
                        try:
                            df.drop(['Prediction'], axis = 1, inplace = True)
                        except Exception as e:
                            self.message_signal.emit(str(e))
                        # df.set_index('Class_Names', inplace = True, drop=True)
                        self.message_signal.emit('Processing... \n Exporting data in excel format to output folder')
                        # QApplication.processEvents()
                        try:
                            sf = StyleFrame(df)
                            sf.set_column_width(columns=df.columns.tolist(), width=20)
                            sf.to_excel(
                                excel_writer = writer, 
                                # best_fit = df.columns.tolist(),
                                sheet_name = imgdir)
                            # writer.save()
                        except Exception as e:
                            print('623', e)
                        self.progress_signal.emit([None,None,len(os.listdir(data_path))-1, num_processed])          
                # time.sleep(1)
                self.progress_signal.emit([None,len(os.listdir(self.imgs_path))-1,None, num_processed])
                print('Finished!')
                self.message_signal.emit('Finished! \nExcel file was exported to output folder: %s'%str(self.output_path))
        else:
            num_total = direct_img_total
                    # print(num_total)
            if num_total < 1:
                self.message_signal.emit("No image in current working folder, please check it！")
                return 
            with StyleFrame.ExcelWriter(path_xls) as writer:
                    data_path=args.path_imgs
                    mm_nms = self.mm_select#[mm_list[i] for (i,x) in enumerate(self.mms) if x == 1]
                    if os.path.isdir(data_path):
                        self.message_signal.emit("Processing...\n Current Image Folder: %s"%data_path)
                        # QApplication.processEvents()
                        tf = transforms.Compose([transforms.Resize((args.size,args.size)),
                                                  transforms.ToTensor(), 
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
                        preds_all = []
                        if self.mm_yesno:
                            meas_all = []
                        vol = 0
                        imgdir = data_path.split('/')[-1]
                        print('path = ', imgdir)
                        self.progress_signal.emit([len(os.listdir(data_path))-1, None, None, num_processed])
                        if self.device_type == 2 or self.image_type == 1:
                            try:
                                print('loading images')
                                fls, imgs = ImLoader(data_path)
                                print('images number:',len(fls))
                                test_data=MyDataset(imgs=imgs, transform= tf) #transforms.ToTensor())        
                                test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)#, num_workers = 4) 
                                img_count = len(test_data)
                                # print("img segmented: %d"%img_count)
                                pred_score=np.zeros((img_count))
                                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                # print("Testing")
                                with torch.no_grad():
                                    # model_cnn.eval()
                                    preds = []
                                    meas = []
                                    for j, imgdata in enumerate(test_loader,0):
                                        imgs_tensor, labels = imgdata
                                        # plot(imgs[0])
                                        batch_x = imgs_tensor.to(device)
                                        outputs = model_cnn(batch_x)
                                        pred = torch.max(outputs, 1)[1]
                                        pred_numpy = pred.cpu().numpy().tolist()
                                        preds += pred_numpy
                                        num_processed += args.batch_size
                                        self.progress_signal.emit([None, 0, (j-1)*args.batch_size, num_processed])
                                        self.message_signal.emit("Processing...\n Current Image Folder: %s \n Image: %d"%(data_path, j*args.batch_size))                                        
                                        # self.message_signal.emit(pred_names)
                                    for k, pred_name in enumerate(preds):
                            
                                        try: 
                                            if self.export_yesno == 1:
                                                if int(pred_name) > -1: # or int(pred_name)== 3:
                                                    new_path = os.path.join(self.export_path, imgdir,  classes[int(pred_name)]) 
                                                    if not os.path.exists(new_path):
                                                        os.makedirs(new_path)
                                                    shutil.copy(fls[k], new_path)       
                                            if self.mm_yesno: 
                                                if int(pred_name) > -1:
                                                    self.message_signal.emit("Measuring...\n Current Image Folder: %s \n Image: %d"%(data_path, k)) 
                                                    self.progress_signal.emit([None, 0, k, k])
                                                    mea = mm(imgs[k],self.mms,self.pxsize)
                                                    # print(mea)
                                                    meas.append(mea)
                                                else:
                                                    mea = [np.nan]*self.mms.count(1)
                                                    meas.append(mea)

                                        except Exception as e:
                                            print("Error with image measurement or write:%s"%str(e))
                                            # self._signal.emit([preds, meas, e])
                                    
                                preds_all += preds
                                if self.mm_yesno:
                                    # print(len(meas))
                                    meas_all += meas
                                self.data_signal.emit([preds_all, imgdir])

                            except Exception as e:
                                self.message_signal.emit(str(e))
                                print('715',e)

                        else:
                            for j, fl in enumerate(os.listdir(data_path)):
                                # print(fl)
        #                         # time.sleep(1)
                                if "summary.txt" in fl:
                                    f = open(os.path.join(data_path, fl), "r", encoding="utf-8")               
                                    for line in f:
                                        if "Fluid Volume Imaged:" in line:
                                            vol = float(line.split(":")[1].replace("ml", ""))
                                            print("Volume imaged = ", vol)                                            
                                if is_img(fl):
                                    num_processed += 1
                                    self.progress_signal.emit([None, 0, j-1, num_processed])
                                    self.message_signal.emit("Processing...\n Current Image Folder: %s \n Image: %s"%(imgdir, fl))
                                    QApplication.processEvents()                    
                                    if (fl.split(".")[-1] in IMG_FORMATS) & ('cal_image' not in fl):
                                        # rintrint('mark1')
                                        try:
                                            if self.device_type == 0:
                                                imgs = Flowcam_Cut(os.path.join(data_path, fl))
                                            elif self.device_type == 1:
                                                imgs = Microscope_Cut(os.path.join(data_path, fl))
    
                                            test_data=MyDataset(imgs=imgs, transform= tf) #transforms.ToTensor())        
                                            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)#, num_workers = 4) 
                                            img_count = len(test_data)
                                            # print("img segmented: %d"%img_count)
                                            # self.message_signal.emit(img_count)
                                            pred_score=np.zeros((img_count))
                                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                            # print("Testing")
                                            with torch.no_grad():
                                                # model_cnn.eval()
                                                preds = []
                                                meas = []
                                                for _, imgdata in enumerate(test_loader,0):
                                                    imgs_tensor, labels = imgdata
                                                    # plot(imgs[0])
                                                    batch_x = imgs_tensor.to(device)
                                                    outputs = model_cnn(batch_x)
                                                    pred = torch.max(outputs, 1)[1]
                                                    pred_numpy = pred.cpu().numpy().tolist()
                                                    preds += pred_numpy
                                                    # self.message_signal.emit(pred_names)
                                                for k, pred_name in enumerate(preds):
                                        
                                                    try: 
                                                        if self.export_yesno == 1:
                                                            if int(pred_name) > -1: # or int(pred_name)== 3:
                                                                new_path = os.path.join(self.export_path, imgdir, classes[int(pred_name)]) 
                                                                if not os.path.exists(new_path):
                                                                    os.makedirs(new_path)
                                                                cv2.imwrite(os.path.join(new_path, fl.split('.')[0]+'_%s_ROI_%06d.png'%(imgdir,k)), imgs[k])   
                                                        if self.mm_yesno: 
                                                            if int(pred_name) > -1:
                                                                # self.message_signal.emit("Measuring...\n Current Image Folder: %s \n Image: %d"%(data_path, k)) 
                                                                # self.progress_signal.emit([None, 0, k, k])
                                                                mea = mm(imgs[k],self.mms,self.pxsize)
                                                                # print('%d:%d'%(i,pred_name))
                                                                # print(mea)
                                                                meas.append(mea)
                                                            else:
                                                                mea = [np.nan]*self.mms.count(1)
                                                                # print(mea)
                                                                meas.append(mea)
                                                                # print('%d:%d'%(i,pred_name))
                                                        # self._signal.emit([preds, meas, None])
                                                        
                                                    except Exception as e:
                                                        print("Error with image measurement or write:%s"%str(e))
                                                        # self._signal.emit([preds, meas, e])
                                                
                                            preds_all += preds
                                            # print("mark4")
            #                                     # print(len(preds_all))
                                            if self.mm_yesno:
                                                meas_all += meas
                                                print(len(preds),len(meas))
            #                                         # print(len(meas))
                                            self.data_signal.emit([preds_all, imgdir])
                                            # QApplication.processEvents()
                                            # print("mark5")
                                        except Exception as e:
                                            self.message_signal.emit(str(e))
                                            print('804',e)
                                            # QApplication.processEvents()
    
                        count = []
                        if self.mm_yesno:
                            df_meas = pd.DataFrame(meas_all,columns=mm_nms)
                            df_meas.replace([np.inf, -np.inf], np.nan, inplace=True)
                            print("length of morphology measures: %d"%len(df_meas))
                            df_meas['Prediction'] = preds_all
                            df = pd.DataFrame(columns = mm_nms+['Prediction'])
                            for n in range(args.features):
                                df_i = df_meas[df_meas["Prediction"] == n].mean()
                                df.loc[n] = df_i             
                                count.append(preds_all.count(n))
                            if vol > 0:
                                df['Counts(ind/ml)'] = [float(x)/vol for x in count]
                                if 'Counts(/ind)' in mm_nms:
                                    df['Counts(cells/ml)'] = df['Counts(ind/ml)'] * df['Counts(/ind)']
                                if 'Total_vol(/ind)' in mm_nms:
                                    df['Total_vol(/ml)'] = df['Counts(ind/ml)'] * df['Total_vol(/ind)']  
                            else:
                                df['Counts(ind)'] = count
                                if 'Counts(/ind)' in mm_nms:
                                    df['Counts(cells)'] = df['Counts(ind)'] * df['Counts(/ind)']
                                if 'Total_vol(/ind)' in mm_nms:
                                    df['Total_vol'] = df['Counts(ind)'] * df['Total_vol(/ind)']  
                                    
                        else:
                            for m in range(args.features):           
                                count.append(preds_all.count(m))
                        
                            if vol > 0:
                                nm = 'Counts(ind/ml)'
                                df = pd.DataFrame([float(x)/vol for x in count], columns=[nm])
                            else:
                                nm = 'Counts(ind)'
                                df = pd.DataFrame(count, columns=[nm])
                        df.insert(0, 'Class_Names', cn_df['Class_Names'])
                        try:
                            df.drop(['Prediction'], axis = 1, inplace = True)
                        except Exception as e:
                            self.message_signal.emit(str(e))
                        # df.set_index('Class_Names', inplace = True, drop=True)
                        self.message_signal.emit('Processing... \n Exporting data in excel format to output folder')
                        # QApplication.processEvents()
                        try:
                            sf = StyleFrame(df)
                            sf.set_column_width(columns=df.columns.tolist(), width=20)
                            sf.to_excel(
                                excel_writer = writer, 
                                # best_fit = df.columns.tolist(),
                                sheet_name = data_path.split('/')[-1])
                            # writer.save()
                        except Exception as e:
                            print('860',e)
                        # time_used = time.time() - start
                        # time_left = int(time_used/num_processed * (num_total - num_processed))
                        self.progress_signal.emit([None,None,len(os.listdir(data_path))-1, num_processed])
                        # time.sleep(1)
            
            # time.sleep(1)
            self.progress_signal.emit([None,len(os.listdir(self.imgs_path))-1,None, num_processed])
            print('Finished!')
            self.message_signal.emit('Finished! \nExcel file was exported to output folder: %s'%str(self.export_path))
    
#%%
# -----------------ready the dataset--------------------------
def default_loader(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def im_file_loader(im_path):
    return cv2.imread(im_path, cv2.IMREAD_COLOR)

def ImLoader(path):
    imgs = []
    fls = []
    if os.path.isdir(path):
        for fl in os.listdir(path):
            im_path = os.path.join(path,fl)
            
            if not os.path.isdir(im_path):
                try:
                    if is_img(im_path):
                        img  = im_file_loader(im_path)
                        imgs.append(img)
                        fls.append(im_path)
                except Exception as e:
                    print('928', e)
                    
    return fls, imgs

class MyDataset(Dataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        img_cv2 = self.imgs[index]
        img = self.loader(img_cv2)
        if self.transform is not None:
            img = self.transform(img)
        return img, index
    def __len__(self):
        return len(self.imgs)

#%%
class Ui_sorta(QMainWindow):
    def __init__(self):
        super(Ui_sorta, self).__init__()
        self.dataset_paths = []
        self.dataset_path = r"E:/"
        self.output_paths = []
        self.output_path = r"E:/Outputs" #os.path.join(GetDesktopPath(),'Outputs')
        self.export_paths = []
        self.export_path =r"E:/Outputs"
        self.formula_paths =[]
        self.formula_path =""
        modelnames = os.listdir(r"./Models")
        self.modeltypes = [d for d in modelnames if os.path.isdir(os.path.join(r"./Models", d))]
        # self.modeltypes = ["Freshwater", "Xiamen","Taiwan Strait","South China Sea",
        #                   "West Pacific","Freshwater Zooplankton", "Marine Zooplankton", "IPP(Dongshan)", "IPP(Fecal Pellet)"]
        
        self.modeltype = self.modeltypes[0]
        self.device_type = 0

        self.image_types = ["original","segmented"]
        self.image_type = 0
        
        # self.loaded_full_model = ""
        # self.loaded_full_models = []
        # self.loaded_state_dict = ""
        # self.loaded_state_dicts = []
        
        self.level = 1
        self.mm_yesno = False
        self.Image_Size = 224
        
        self.loaded_class_path = ""
        self.taxa_list = ""
        self.taxa_num = 0

        self.image_path = ""
        self.image_list = []

        self.pxsize = 0.485
        
        self.features = 100
        
        self.export_yesno = 0
        
        self.time_used = 0
        self.time_left = 0
        # self.preds = []
        # self.meas = []
#%% morphology parameters settings
        self.mms = [0,0,0,0,0,0,0,0,0,0,0]
        self.mm_select =[]
        self.mm_list = ["Length","Width","Diameter","Area","Perimeter","Length_ellipse", "Width_ellipse","area_ellipse", "Vol_ellipse", "Counts(/ind)", "Total_vol(/ind)"]
               
#%% Setup UI
        
        self.setupUi()

#%%
    def setupUi(self):
        self.setObjectName("sorta")
        self.setWindowTitle('sorta: GUI for AI plankton recognization')
        desktop = QApplication.desktop()
        # print(desktop.width(), desktop.height())
        self.resize(desktop.width(), desktop.height())
        self.setMinimumSize(QSize(1200, 800))
        self.setMaximumSize(QSize(4000, 2400))
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.setFont(font)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_top = QFrame(self.centralwidget)
        self.gridLayout.addWidget(self.frame_top, 0, 0, 1, 1)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_top.sizePolicy().hasHeightForWidth())
        self.frame_top.setSizePolicy(sizePolicy)
        self.frame_top.setMinimumSize(QSize(0, 100))
        self.frame_top.setFrameShape(QFrame.StyledPanel)
        self.frame_top.setFrameShadow(QFrame.Raised)
        self.frame_top.setObjectName("frame")
        self.horizontalLayout = QHBoxLayout(self.frame_top)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 10)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QLabel(self.frame_top)
        self.label.setMaximumSize(QSize(280, 100))
        # self.label.setText("brigchen@xmu.edu.cn")
        self.label.setPixmap(QPixmap("sorta_3_long.png"))
        self.label.setScaledContents(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.frame_top_right = QFrame(self.frame_top)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_top_right.sizePolicy().hasHeightForWidth())
        self.frame_top_right.setSizePolicy(sizePolicy)
        self.frame_top_right.setFrameShape(QFrame.StyledPanel)
        self.frame_top_right.setFrameShadow(QFrame.Raised)
        self.frame_top_right.setObjectName("frame_4")
        self.gridLayout_top = QGridLayout(self.frame_top_right)
        self.gridLayout_top.setContentsMargins(100, 0, 100, 0)
        self.gridLayout_top.setHorizontalSpacing(100)
        self.gridLayout_top.setVerticalSpacing(0)
        self.gridLayout_top.setObjectName("gridLayout_top")
        
        # self.pushButton_seg = QPushButton(self.frame_top_right)
        # # self.pushButton_seg.setObjectName("sorta")
        # self.pushButton_seg.clicked.connect(self.on_start_sorta)
        # self.gridLayout_top.addWidget(self.pushButton_seg, 0, 0, 1, 1)
        
        # self.pushButton_tag = QPushButton(self.frame_top_right)
        # # self.pushButton_tag.setObjectName("sorta")
        # self.pushButton_tag.clicked.connect(self.on_start_tagor)
        # self.gridLayout_top.addWidget(self.pushButton_tag, 0, 1, 1, 1)
        
        # self.pushButton_mod = QPushButton(self.frame_top_right)
        # # self.pushButton_mod.setObjectName("sorta")
        # self.pushButton_mod.clicked.connect(self.on_start_sorta)
        # self.gridLayout_top.addWidget(self.pushButton_mod, 0, 2, 1, 1)
        
        # self.pushButton_sor = QPushButton(self.frame_top_right)
        # # self.pushButton_sor.setObjectName("sorta")
        # self.pushButton_sor.clicked.connect(self.on_start_sorta)
        # self.gridLayout_top.addWidget(self.pushButton_sor, 0, 3, 1, 1)
        
        self.pushButton_start = QPushButton(self.frame_top_right)
        self.pushButton_start .setMinimumHeight(50)
        self.pushButton_start.setText('Start')
        self.pushButton_start.setFont(QFont("Arial", 18))
        # self.pushButton_start.setObjectName("dataset")
        self.pushButton_start.clicked.connect(self.on_start_iden)
        self.gridLayout_top.addWidget(self.pushButton_start, 1, 0, 1, 1)    
                       
        self.pushButton_exit = QPushButton(self.frame_top_right)
        self.pushButton_exit.setMinimumHeight(50)
        self.pushButton_exit.setFont(QFont("Arial", 18))
        # self.pushButton_exit.setObjectName("start")
        self.pushButton_exit.setText('Exit')
        self.pushButton_exit.clicked.connect(self.close)
        self.gridLayout_top.addWidget(self.pushButton_exit, 1, 2, 1, 1)
        
        self.horizontalLayout.addWidget(self.frame_top_right)
       
        self.frame_down = QFrame(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.frame_down.sizePolicy().hasHeightForWidth())
        self.frame_down.setSizePolicy(sizePolicy)
        self.frame_down.setFrameShape(QFrame.StyledPanel)
        self.frame_down.setFrameShadow(QFrame.Raised)
        self.frame_down.setObjectName("frame_down")
        
        self.horizontalLayout_down = QHBoxLayout(self.frame_down)
        self.horizontalLayout_down.setContentsMargins(0, 0, 0, 10)
        self.horizontalLayout_down.setSpacing(0)
        self.horizontalLayout_down.setObjectName("horizontalLayout_down")
        
        self.frame_down_left = QFrame(self.frame_down)
        self.frame_down_left.setMinimumSize(QSize(250, 0))
        self.frame_down_left.setMaximumSize(QSize(350, 16777215))
        self.frame_down_left.setFrameShape(QFrame.StyledPanel)
        self.frame_down_left.setFrameShadow(QFrame.Raised)
        self.frame_down_left.setObjectName("frame_down_left")
        self.horizontalLayout_down.addWidget(self.frame_down_left)
        
        self.verticalLayout_left = QVBoxLayout(self.frame_down_left)
        self.verticalLayout_left.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_left.setSpacing(10)
        
        self.groupBox_dataset = QGroupBox(self.frame_down_left)
        self.groupBox_dataset.setMinimumSize(QSize(0, 100))
        self.groupBox_dataset.setObjectName("dataset")
        self.verticalLayout_left.addWidget(self.groupBox_dataset)
        self.verticalLayout_dataset = QVBoxLayout(self.groupBox_dataset)
        
        self.pushButton_dataset = QPushButton(self.groupBox_dataset)
        self.pushButton_dataset.setObjectName("dataset")
        self.verticalLayout_dataset.addWidget(self.pushButton_dataset)
        self.pushButton_dataset.clicked.connect(self.open_dataset_folder)
        
        self.comboBox_dataset = QComboBox(self.groupBox_dataset)
        self.comboBox_dataset.setObjectName("dataset_cb")
        self.comboBox_dataset.currentIndexChanged.connect(self.renew_dataset_folder) 
        self.verticalLayout_dataset.addWidget(self.comboBox_dataset)


        self.groupBox_output = QGroupBox(self.frame_down_left)
        self.groupBox_output.setMinimumSize(QSize(0, 100))
        self.groupBox_output.setObjectName("output")
        self.verticalLayout_left.addWidget(self.groupBox_output)

        self.verticalLayout_output = QVBoxLayout(self.groupBox_output)
        self.verticalLayout_output.setObjectName("verticalLayout_output")
        self.pushButton_output = QPushButton(self.groupBox_output)
        self.pushButton_output.setObjectName("output_btn")
        self.verticalLayout_output.addWidget(self.pushButton_output)
        self.pushButton_output.clicked.connect(self.open_output_folder)
        
        self.comboBox_output = QComboBox(self.groupBox_output)
        self.comboBox_output.setObjectName("output_cb")
        self.comboBox_output.currentIndexChanged.connect(self.renew_output_folder) 
        self.verticalLayout_output.addWidget(self.comboBox_output)

        self.groupBox_mode = QGroupBox(self.frame_down_left)                        #Settings
        self.verticalLayout_left.addWidget(self.groupBox_mode)
        self.verticalLayout_mode = QVBoxLayout(self.groupBox_mode)

        self.groupBox_31 = QGroupBox(self.groupBox_mode)    #Image types
        self.groupBox_31.setObjectName("Model types")
        self.groupBox_31.setMinimumSize(QSize(200, 50))
        self.groupBox_31.setMaximumSize(QSize(500, 100))
        self.verticalLayout_mode.addWidget(self.groupBox_31)
        
        self.comboBox_mtype=QComboBox(self.groupBox_31)
        self.comboBox_mtype.addItems(self.modeltypes)
        self.comboBox_mtype.setCurrentIndex(0)
        self.comboBox_mtype.currentIndexChanged.connect(self.on_mtype)
        self.verticalLayout_mtype = QVBoxLayout(self.groupBox_31)
        self.verticalLayout_mtype.addWidget(self.comboBox_mtype)

        self.groupBox_32 = QGroupBox(self.groupBox_mode)    #min size
        self.groupBox_32.setMinimumSize(QSize(200, 75))
        self.groupBox_32.setMaximumSize(QSize(500, 150))
        self.verticalLayout_mode.addWidget(self.groupBox_32)
        self.radioButton_flowcam = QRadioButton(self.groupBox_32)
        self.radioButton_flowcam.setObjectName("flowcam")
        self.radioButton_flowcam.setText("FlowCAM")
        self.radioButton_microscope = QRadioButton(self.groupBox_32)
        self.radioButton_microscope.setObjectName("microscope")
        self.radioButton_microscope.setText("Microscope")
        self.radioButton_ipp = QRadioButton(self.groupBox_32)
        self.radioButton_ipp.setObjectName("ipp")
        self.radioButton_ipp.setText("IPP")
        self.radioButton_flowcam.setChecked(True)
        self.verticalLayout_device_type = QVBoxLayout(self.groupBox_32)
        self.verticalLayout_device_type.addWidget(self.radioButton_flowcam)
        self.verticalLayout_device_type.addWidget(self.radioButton_microscope)
        self.verticalLayout_device_type.addWidget(self.radioButton_ipp)
        self.radioButton_flowcam.toggled.connect(self.on_device_type)
        self.radioButton_microscope.toggled.connect(self.on_device_type)

        self.groupBox_35 = QGroupBox(self.groupBox_mode)    #magnification
        self.groupBox_35.setTitle("Image types")
        self.verticalLayout_mode.addWidget(self.groupBox_35)
        self.horizonalLayout_35 = QHBoxLayout(self.groupBox_35)
        self.radioButton_single = QRadioButton(self.groupBox_35)
        self.radioButton_single.setText("Original")
        self.radioButton_single.setChecked(True)
        self.radioButton_multi = QRadioButton(self.groupBox_35)
        self.radioButton_multi.setText("Segmented")
        # self.radioButton_multi.setEnabled(False)
        self.horizonalLayout_35.addWidget(self.radioButton_single)
        self.horizonalLayout_35.addWidget(self.radioButton_multi)
        self.radioButton_single.toggled.connect(lambda: self.on_imagetype(0))
        self.radioButton_multi.toggled.connect(lambda: self.on_imagetype(1)) 

        self.groupBox_37 = QGroupBox(self.groupBox_mode)    #magnification
        self.groupBox_37.setTitle("Directory Level")
        # self.groupBox_35.setMinimumSize(QSize(200, 50))
        # self.groupBox_35.setMaximumSize(QSize(500, 100))
        self.verticalLayout_mode.addWidget(self.groupBox_37)
        # self.groupBox_37.setSizePolicy(sizePolicy)
        self.radioButton_level0 = QRadioButton(self.groupBox_37)
        self.radioButton_level0.setObjectName("level0")
        self.radioButton_level0.setText("Current directory")
        self.radioButton_level1 = QRadioButton(self.groupBox_37)
        self.radioButton_level1.setObjectName("level1")
        self.radioButton_level1.setText("Sub-directories")
        self.radioButton_level1.setChecked(True)
        self.verticalLayout_level = QVBoxLayout(self.groupBox_37)
        self.verticalLayout_level.addWidget(self.radioButton_level0)
        self.verticalLayout_level.addWidget(self.radioButton_level1)
        self.radioButton_level0.toggled.connect(self.on_level)
        self.radioButton_level1.toggled.connect(self.on_level)

        self.groupBox_39 = QGroupBox(self.groupBox_mode)    #magnification
        self.groupBox_39.setTitle("Classified Images Export")
        self.verticalLayout_mode.addWidget(self.groupBox_39)
        self.groupBox_export_yesno = QGroupBox(self.groupBox_39)    #magnification
        self.horizonalLayout_export_yesno = QHBoxLayout(self.groupBox_export_yesno)
        self.radioButton_export_no = QRadioButton(self.groupBox_export_yesno)
        self.radioButton_export_no.setText("No")
        self.radioButton_export_no.setChecked(True)
        self.radioButton_export_yes = QRadioButton(self.groupBox_export_yesno)
        self.radioButton_export_yes.setText("Yes")
        self.radioButton_export_yes.setChecked(False)
        self.horizonalLayout_export_yesno.addWidget(self.radioButton_export_no)
        self.horizonalLayout_export_yesno.addWidget(self.radioButton_export_yes)
        self.radioButton_export_no.toggled.connect(lambda: self.on_export(0))
        self.radioButton_export_yes.toggled.connect(lambda: self.on_export(1))

        # self.groupBox_export = QGroupBox(self.groupBox_mode)
        # # self.groupBox_.setMinimumSize(QSize(0, 100))
        # self.groupBox_export.setObjectName("export folder")
        # self.verticalLayout_mode.addWidget(self.groupBox_export)

        self.verticalLayout_39 = QVBoxLayout(self.groupBox_39)
        self.verticalLayout_39.addWidget(self.groupBox_export_yesno)
        # self.verticalLayout_output.setObjectName("verticalLayout_output")
        self.pushButton_export = QPushButton(self.groupBox_39)
        self.pushButton_export.setText("Image Export Folder")
        self.pushButton_export.setEnabled(False)
        # self.pushButton_export.setObjectName("output_btn")
        self.verticalLayout_39.addWidget(self.pushButton_export)
        self.pushButton_export.clicked.connect(self.open_export_folder)
        
        self.comboBox_export = QComboBox(self.groupBox_39)
        # self.comboBox_export.setObjectName("output_cb")
        self.comboBox_export.currentIndexChanged.connect(self.renew_export_folder) 
        self.verticalLayout_39.addWidget(self.comboBox_export)
        
        
        self.groupBox_38 = QGroupBox(self.groupBox_mode)    #magnification
        # self.groupBox_37.setTitle("Image Size Setup")
        self.verticalLayout_mode.addWidget(self.groupBox_38)
        self.groupBox_38.setSizePolicy(sizePolicy)
        
        self.frame_down_mid = QFrame(self.frame_down)
        self.frame_down_mid.setMinimumSize(QSize(250, 0))
        self.frame_down_mid.setMaximumSize(QSize(350, 16777215))
        self.frame_down_mid.setFrameShape(QFrame.StyledPanel)
        self.frame_down_mid.setFrameShadow(QFrame.Raised)
        self.frame_down_mid.setObjectName("frame_down_mid")
        self.horizontalLayout_down.addWidget(self.frame_down_mid)
        self.verticalLayout_morph = QVBoxLayout(self.frame_down_mid)
        self.verticalLayout_morph.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_morph.setSpacing(10)

        self.groupBox_morph = QGroupBox(self.frame_down_mid)                        #
        self.groupBox_morph.setTitle("Plankton Morphology")
        self.groupBox_morph.setSizePolicy(sizePolicy)
        self.verticalLayout_morph.addWidget(self.groupBox_morph)
        self.verticalLayout_mm1 = QVBoxLayout(self.groupBox_morph)

        self.groupBox_41 = QGroupBox(self.groupBox_morph)
        self.groupBox_41.setTitle("Morp Measurement")
        self.verticalLayout_mm1.addWidget(self.groupBox_41)
        self.radioButton_mmyes = QRadioButton(self.groupBox_41)
        self.radioButton_mmyes.setText("Yes")
        self.radioButton_mmno = QRadioButton(self.groupBox_41)
        self.radioButton_mmno.setText("No")
        self.radioButton_mmno.setChecked(True)
        self.HorizonalLayout_41 = QHBoxLayout(self.groupBox_41)
        self.HorizonalLayout_41.addWidget(self.radioButton_mmyes)
        self.HorizonalLayout_41.addWidget(self.radioButton_mmno)
        self.radioButton_mmyes.toggled.connect(lambda: self.on_mm_yesno(0))
        self.radioButton_mmno.toggled.connect(lambda: self.on_mm_yesno(1))

        self.groupBox_36 = QGroupBox(self.groupBox_morph)    #magnification
        self.groupBox_36.setTitle("Pixel Size Setup")
        self.verticalLayout_mode.addWidget(self.groupBox_36)
        self.horizonalLayout_36 = QHBoxLayout(self.groupBox_36)
        self.label_size = QtWidgets.QLabel(self.groupBox_36)

        self.label_size.setText("Pixel Size (um):")
        self.lineedit_size = QtWidgets.QLineEdit(self.groupBox_36)
        self.groupBox_36.setEnabled(False)
        # self.pixelsize.setEnabled(True)
        # intValidator = QIntValidator(self)
        # intValidator.setRange(10,1000)
        # self.lineedit_size.setValidator(intValidator)
        self.lineedit_size.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit_size.setText("%.3f"%self.pxsize)
        self.lineedit_size.editingFinished.connect(self.size_changed)
       
        self.horizonalLayout_36.addWidget(self.label_size)
        self.horizonalLayout_36.addWidget(self.lineedit_size)
        self.verticalLayout_mm1.addWidget(self.groupBox_36)

        self.label_px_hint = QLabel(self.groupBox_morph)
        self.label_px_hint.setFont(QFont("Roman times",9))
        self.label_px_hint.setText('*   FlowCAM 10X: 0.485 4X: 1.21\n    IPP 0.2X: 19.3  0.5X: 7.72\n**   Manual measure for other lens or microscopes!\n*** Use 1.0 if pixelsize not available\n')
        self.verticalLayout_mm1.addWidget(self.label_px_hint)

        self.groupBox_42 = QGroupBox(self.groupBox_morph)
        self.groupBox_42.setTitle("Basic")
        self.groupBox_42.setEnabled(False)
        self.verticalLayout_mm1.addWidget(self.groupBox_42)
        self.gridLayout_42 = QGridLayout(self.groupBox_42)
        self.checkbox_1 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_1.setText("Length")
        self.checkbox_1.setChecked(False)
        self.checkbox_1.stateChanged.connect(lambda: self.on_mm_select(1))
        self.checkbox_2 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_2.setText("Width")
        self.checkbox_2.setChecked(False)
        self.checkbox_2.stateChanged.connect(lambda: self.on_mm_select(2))
        self.checkbox_3 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_3.setText("Eq. Diameter")
        self.checkbox_3.setChecked(False)
        self.checkbox_3.stateChanged.connect(lambda: self.on_mm_select(3))
        self.checkbox_4 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_4.setText("Area")
        self.checkbox_4.setChecked(False)
        self.checkbox_4.stateChanged.connect(lambda: self.on_mm_select(4))
        self.checkbox_5 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_5.setText("Perimeter")
        self.checkbox_5.setChecked(False)
        self.checkbox_5.stateChanged.connect(lambda: self.on_mm_select(5))
        self.checkbox_6 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_6.setText("Length-ellipse fit")
        self.checkbox_6.setChecked(False)
        self.checkbox_6.stateChanged.connect(lambda: self.on_mm_select(6))
        self.checkbox_7 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_7.setText("Width-ellipse fit")
        self.checkbox_7.setChecked(False)
        self.checkbox_7.stateChanged.connect(lambda: self.on_mm_select(7))
        self.checkbox_8 = QtWidgets.QCheckBox(self.groupBox_42)
        self.checkbox_8.setText("Area-ellipse fit")
        self.checkbox_8.setChecked(False)
        self.checkbox_8.stateChanged.connect(lambda: self.on_mm_select(8))
        
        self.gridLayout_42.addWidget(self.checkbox_1, 0, 0, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_2, 1, 0, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_3, 2, 0, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_4, 3, 0, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_5, 0, 1, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_6, 1, 1, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_7, 2, 1, 1, 1)
        self.gridLayout_42.addWidget(self.checkbox_8, 3, 1, 1, 1)

        self.groupBox_43 = QGroupBox(self.groupBox_morph)
        self.groupBox_43.setTitle("Biovolume")
        self.groupBox_43.setEnabled(False)
        self.verticalLayout_mm1.addWidget(self.groupBox_43)
        self.verticalLayout_43 = QVBoxLayout(self.groupBox_43)
        self.checkbox_9 = QtWidgets.QCheckBox(self.groupBox_43)
        self.checkbox_9.setText("Ellipse-Fit Volume")
        self.checkbox_9.setChecked(False)
        self.checkbox_9.stateChanged.connect(lambda: self.on_mm_select(9))

        self.checkbox_10 = QtWidgets.QCheckBox(self.groupBox_43)
        self.checkbox_10.setText("Cell numbers")
        self.checkbox_10.setChecked(False)
        self.checkbox_10.stateChanged.connect(lambda: self.on_mm_select(10))
        self.checkbox_11 = QtWidgets.QCheckBox(self.groupBox_43)
        self.checkbox_11.setText("Total Volume(multi-celluar)")
        self.checkbox_11.setChecked(False)
        self.checkbox_11.stateChanged.connect(lambda: self.on_mm_select(11))
        
        self.verticalLayout_43.addWidget(self.checkbox_9)
        self.verticalLayout_43.addWidget(self.checkbox_10)
        self.verticalLayout_43.addWidget(self.checkbox_11)

       
        self.groupBox_morph2 = QGroupBox(self.frame_down_mid)    #
        self.verticalLayout_morph.addWidget(self.groupBox_morph2)
        self.verticalLayout_h2 = QVBoxLayout(self.groupBox_morph2)
        self.groupBox_morph2.setSizePolicy(sizePolicy)

        self.frame_down_mid2 = QFrame(self.frame_down)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down_mid2.sizePolicy().hasHeightForWidth())
        self.frame_down_mid2.setSizePolicy(sizePolicy)
        self.frame_down_mid2.setFrameShape(QFrame.StyledPanel)
        self.frame_down_mid2.setFrameShadow(QFrame.Raised)
        self.frame_down_mid2.setObjectName("frame_6")
        self.horizontalLayout_down.addWidget(self.frame_down_mid2)
        
        self.verticalLayout_view = QVBoxLayout(self.frame_down_mid2)
        self.verticalLayout_view.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_view.setSpacing(0)

        self.groupBox_status = QGroupBox(self.frame_down_mid2)
        self.groupBox_status.setObjectName("status")
        self.verticalLayout_view.addWidget(self.groupBox_status)
        self.groupBox_status.setMinimumHeight(200)
        self.groupBox_status.setMaximumSize(5000,250)
        self.groupBox_status.setTitle("Status")
        self.verticalLayout_status = QVBoxLayout(self.groupBox_status)
        
        self.groupBox_labels = QGroupBox(self.groupBox_status)
        self.verticalLayout_status.addWidget(self.groupBox_labels)
        self.horizontalLayout_labels = QHBoxLayout(self.groupBox_labels)
        self.label_status = QLabel(self.groupBox_labels)
        self.label_status.setText("Welcome to use sorta tool for plankton identification")
        self.label_status.setStyleSheet('color:red')
        self.label_status.setMinimumWidth(650)
        # self.label_status.setFont(QFont(QFont.Bold))
        self.horizontalLayout_labels.addWidget(self.label_status)

        self.groupBox_lcd = QGroupBox(self.groupBox_labels)
        self.groupBox_lcd.hide()
        self.horizontalLayout_labels.addWidget(self.groupBox_lcd)
        self.gridLayout_lcd = QGridLayout(self.groupBox_lcd)
        self.label_lcd_1 = QLabel(self.groupBox_lcd)
        self.label_lcd_1.setText('Time used:')        
        self.label_lcd_2 = QLabel(self.groupBox_lcd)
        self.label_lcd_2.setText('Time estimated:')
        self.lcdNumber1 = QLCDNumber(self.groupBox_lcd)
        self.lcdNumber1.setGeometry(QtCore.QRect(0, 0, 250, 50))
        self.lcdNumber1.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.lcdNumber1.setFrameShape(QFrame.NoFrame)
        self.lcdNumber1.setSmallDecimalPoint(False)
        self.lcdNumber1.setDigitCount(8)
        self.lcdNumber1.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber1.setProperty("value", "00:00:00")
        self.lcdNumber1.display('00:00:00')
        self.lcdNumber2 = QLCDNumber(self.groupBox_lcd)
        self.lcdNumber2.setGeometry(QtCore.QRect(0, 0, 250, 50))
        self.lcdNumber2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.lcdNumber2.setFrameShape(QFrame.NoFrame)
        self.lcdNumber2.setSmallDecimalPoint(False)
        self.lcdNumber2.setDigitCount(8)
        self.lcdNumber2.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber2.setProperty("value", "00:00:00")
        self.lcdNumber2.display('00:00:00')
        self.gridLayout_lcd.addWidget(self.label_lcd_1,0,0,1,1)
        self.gridLayout_lcd.addWidget(self.label_lcd_2,1,0,1,1)
        self.gridLayout_lcd.addWidget(self.lcdNumber1,0,1,1,2)
        self.gridLayout_lcd.addWidget(self.lcdNumber2,1,1,1,2)
               
        self.pbar1 = QProgressBar(self) 
        self.verticalLayout_status.addWidget(self.pbar1)
        self.pbar1.hide()
        
        self.pbar2 = QProgressBar(self)         
        self.verticalLayout_status.addWidget(self.pbar2)
        self.pbar2.hide()

        self.groupBox_plot = QGroupBox(self.frame_down_mid2)
        self.groupBox_plot.setTitle("Visualization")
        self.verticalLayout_view.addWidget(self.groupBox_plot)

        self.frame_plot1 = QFrame(self.groupBox_plot)
        self.verticalLayout_plot = QVBoxLayout(self.groupBox_plot)
        self.verticalLayout_plot.setContentsMargins(0, 0, 0, 0)
        
        # self.figure = plt.figure(figsize=(8,6),dpi = 100)
        self.label_plot = QLabel(self.groupBox_plot)
        self.label_plot.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.hlayout_plot = QHBoxLayout(self.label_plot)
        self.verticalLayout_plot.addWidget(self.label_plot)
        
        img = Image.open("ALGAVISION-01.png")
        self.figure = Figure_Bar(8, 6, dpi=100)
        # self.figure.setGeometry(QtCore.QRect(0, 0, 800, 600))
        # self.plot1 = FigureCanvasQTAgg(self.figure)
        # self.ax = self.figure.add_subplot(111)
        # plt.subplots_adjust(top=.99, bottom=0.25, right=.99, left=.075)
        plt.imshow(img)
        plt.axis('off')
        # plt.show()
        self.hlayout_plot.addWidget(self.figure)
        

        # self.plot1 = QGraphicsView(self.groupBox_plot)
        # self.plot1.setStyleSheet("background:transparent;border:0px;")
        # scene = QGraphicsScene()
        # imgShow = QPixmap()
        # imgShow.load("ALGAVISION-01.png")#Sorta_logo2.png
        # imgShowItem = QGraphicsPixmapItem()
        # imgShowItem.setPixmap(QPixmap(imgShow))
        # imgShowItem.setPixmap(QPixmap(imgShow).scaled(700, 650))
        # scene.addItem(imgShowItem)
        # self.plot1.setScene(scene)
        # print(self.plot1.width(), self.plot1.height())
        # self.myImage = Figure_Bar(8, 6, dpi=100)
        # self.myGraphyScene = QGraphicsScene()
        #取消水平和垂直滚动条
        # self.plot1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        # self.plot1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.myQGraphicsProxyWidget = self.myGraphyScene.addWidget(self.myImage)
        


#%% Down right(last) frame for class name list

        self.frame_down_right = QFrame(self.frame_down)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down_right.sizePolicy().hasHeightForWidth())
        self.frame_down_right.setSizePolicy(sizePolicy)
        self.frame_down_right.setFrameShape(QFrame.StyledPanel)
        self.frame_down_right.setFrameShadow(QFrame.Raised)
        self.frame_down_right.setMaximumSize(QSize(300, 16777215))

        self.horizontalLayout_down.addWidget(self.frame_down_right)
        self.verticalLayout_right = QVBoxLayout(self.frame_down_right)
        self.verticalLayout_right.setContentsMargins(0, 0, 0, 0)

        self.groupBox_classname = QGroupBox(self.frame_down_right)
        self.groupBox_classname.setTitle("Counts of Classes")
        self.verticalLayout_right.addWidget(self.groupBox_classname)
        self.verticalLayout_classname = QVBoxLayout(self.groupBox_classname)
        
        self.pushButton_load_class = QPushButton(self.groupBox_classname)
        self.pushButton_load_class.setText("Load class names")
        self.verticalLayout_classname.addWidget(self.pushButton_load_class)
        self.pushButton_load_class.clicked.connect(self.load_class)
        
                
        self.label_class_num = QLabel(self.groupBox_classname)
        self.label_class_num.setText("Taxa number: %d"%self.taxa_num)
        self.verticalLayout_classname.addWidget(self.label_class_num)

        self.listview_class = QtWidgets.QTableView(self.groupBox_classname)
        self.listview_class.setMaximumSize(QSize(300, 16777215))
        self.listview_class.setObjectName("class list")
        self.update_class()

        self.verticalLayout_classname.addWidget(self.listview_class)
        
        self.gridLayout.addWidget(self.frame_down, 1, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)
        
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.setMenuBar(self.menubar)
        
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        
        self.actionExit = QAction(self)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.close_app)

        self.actionOpen_Dataset_Folder = QAction(self)
        self.actionOpen_Dataset_Folder.setObjectName("actionOpen_Dataset_Folder")
        self.actionOpen_Dataset_Folder.triggered.connect(self.open_dataset_folder)
        self.actionOpen_Output_Folder = QAction(self)
        self.actionOpen_Output_Folder.setObjectName("actionOpen_Output_Folder")
        self.actionOpen_Output_Folder.triggered.connect(self.open_output_folder)

        self.actionHelp = QAction(self)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.triggered.connect(self.on_help)
        
        self.actionVersion = QAction(self)
        self.actionVersion.setObjectName("actionVersion")
        self.actionVersion.triggered.connect(self.on_version)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Dataset_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Output_Folder)
        self.menuFile.addSeparator()

        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionVersion)
        self.menuHelp.addAction(self.actionHelp)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi()
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("sorta", "sorta"))
        
        self.groupBox_dataset.setTitle(_translate("sorta", "Plankton Images Folder"))
        self.pushButton_dataset.setText(_translate("sorta", "Open Plankton Images Folder"))
        self.groupBox_output.setTitle(_translate("sorta", "Output Folder"))
        self.pushButton_output.setText(_translate("sorta", "Open Output Folder"))
        
        self.groupBox_mode.setTitle(_translate("sorta", "Model Settings"))
        self.groupBox_31.setTitle(_translate("sorta", "Models for Specific Location"))
        self.groupBox_32.setTitle(_translate("sorta", "Imaging Device Type"))


        self.menuFile.setTitle(_translate("sorta", "File"))
        self.menuHelp.setTitle(_translate("sorta", "Help"))
        self.actionExit.setText(_translate("sorta", "Exit"))
        self.actionOpen_Dataset_Folder.setText(_translate("sorta", "Open Images Folder"))
        self.actionOpen_Output_Folder.setText(_translate("sorta", "Open Output Folder"))

        self.actionHelp.setText(_translate("sorta", "Help"))
        self.actionVersion.setText(_translate("sorta", "Version"))
        self.load_config()
        
    # def plotting(img):
    #     scene = QGraphicsScene()
    #     imgShow = QPixmap()
    #     imgShow.load("ALGAVISION-01.png")#Sorta_logo2.png
    #     imgShowItem = QGraphicsPixmapItem()
    #     # imgShowItem.setPixmap(QPixmap(imgShow))
    #     imgShowItem.setPixmap(QPixmap(imgShow).scaled(800, 800))
    #     scene.addItem(imgShowItem)

    def load_config(self):
        print('------------load config-------------')
        config = configparser.ConfigParser()
        if os.path.exists('sorta.cfg'):
            try:
                config.read('sorta.cfg')
                self.dataset_path = config.get('Settings', 'dataset_path')
                if not check_path(self.dataset_path):
                    self.label_status.setText(u'文件夹名或图像文件名中具有中文字符或特殊符号，请改成纯英文名称！')
                    self.dataset_path = ''
                self.output_path = config.get('Settings', 'output_path')
                self.export_path = config.get('Settings', 'export_path')
                self.modeltype = config.get('Settings', 'modeltype')
                self.device_type = int(config.get('Settings', 'device_type'))
                self.image_type = int(config.get('Settings', 'image_type'))
                self.level = int(config.get('Settings', 'level'))
                self.mm_yesno = eval(config.get('Settings', 'mm_yesno'))
                self.pxsize = float(config.get('Settings', 'pxsize'))
                self.export_yesno = int(config.get('Settings', 'export_yesno'))
                mms = eval(config.get('Settings', 'mms'))
              
                # dataset_path
                self.dataset_paths.append(self.dataset_path)
                self.comboBox_dataset.addItem(self.dataset_path.split("/")[-1])
                self.comboBox_dataset.setCurrentText(self.dataset_path.split("/")[-1])
                
                # output_path
                self.output_paths.append(self.output_path)
                self.comboBox_output.addItem(self.output_path.split("/")[-1])
                self.comboBox_output.setCurrentText(self.output_path.split("/")[-1])                
                
                # export_path
                self.export_paths.append(self.export_path)
                self.comboBox_export.addItem(self.export_path.split("/")[-1])
                self.comboBox_export.setCurrentText(self.export_path.split("/")[-1])  
                
                # modeltype
                self.comboBox_mtype.setCurrentText(self.modeltype)
                self.update_class()
                self.update_parameter()
                
                # image_type
                if self.image_type == 0:
                    self.radioButton_single.setChecked(True)
                else:
                    self.radioButton_multi.setChecked(True)

                # device_type - 根据配置设置单选按钮  
                if self.device_type == 0:  
                    self.radioButton_flowcam.setChecked(True)  
                elif self.device_type == 1:  
                    self.radioButton_microscope.setChecked(True)  
                elif self.device_type == 2:  
                    self.radioButton_ipp.setChecked(True)  
                # =================================== 

                # level_type
                if self.level == 0:
                    self.radioButton_level0.setChecked(True)
                else:
                    self.radioButton_level1.setChecked(True)
                    
                # mm_yesno
                if not self.mm_yesno:
                    self.radioButton_mmno.setChecked(True)
                else:
                    self.radioButton_mmyes.setChecked(True)
                    
                # export_yesno
                if self.export_yesno == 0:
                    self.radioButton_export_no.setChecked(True)
                else:
                    self.radioButton_export_yes.setChecked(True)
                    
                # pxsize
                self.lineedit_size.setText(str(self.pxsize))
                
                # load mms
                for i, yn in enumerate(mms):
                    self.mms[i] = yn
                    eval('self.checkbox_%d.setChecked(%d)'%(i+1, yn))
                    
            except Exception as e:
                print(u'导入配置参数失败:', e)
                self.label_status.setText(u'导入配置参数失败')

                
    def save_config(self):
        print('1695: save')
        config = configparser.ConfigParser()
        config['Settings'] = {
                'dataset_path': self.dataset_path,
                'output_path': self.output_path, 
                'export_path': self.export_path,
                'modeltype': self.modeltype,
                'device_type': self.device_type,
                'image_type': self.image_type,
                'level': self.level,
                'mm_yesno': self.mm_yesno,
                'pxsize': self.pxsize,
                'export_yesno': self.export_yesno,
                'mms': self.mms,
        }
        with open('sorta.cfg', 'w') as configfile:
            config.write(configfile)

    def close_app(self):
        self.save_config()
        self.close()
        
    def update_plot(self, x,y,title):
        self.figure.showImage(x,y, title)
        
        # self.ax.clear() #每次清空绘图
        # self.ax.bar(x, y, width = 0.8, align = 'center', label = title) 
        # # self.myAxes.set_title(title, fontsize)
        # self.ax.set_ylabel('Counts')
        # self.ax.set_xlabel("Taxa")
        # self.ax.title.set_size(20)
        # self.ax.xaxis.label.set_size(10)
        # self.ax.yaxis.label.set_size(10)
        # self.ax.tick_params(axis='x', labelsize= 10, rotation = 90)
        # self.ax.tick_params(axis='y', labelsize= 10)
        # # plt.legend(loc='upper right', fontsize=20)#, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
        #    # ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
        # # self.fig.canvas.draw()  
        # # x = linspace(-3,3,6000)
        # # y = eval(self.line.text()) #使用了eval函数
        # # ax.plot(x,y)
        # self.plot1.draw()
        
    # def lcd_countdown(self):
    #     # time_format = QTime.currentTime()
    #     # time_format = time_format.toString("hh:mm:ss")
    #     self.time_left = self.time_left - 1 if self.time_left > 1 else 0
    #     self.time_used = self.time_used + 1 if self.time_left > 1 else self.time_used
    #     time_format1 = "{:02.0f}:{:02.0f}:{:02.0f}".format(self.time_used // 3600, 
    #                                            (self.time_used % 3600) // 60, 
    #                                             self.time_used % 60)  
    #     time_format2 = "{:02.0f}:{:02.0f}:{:02.0f}".format(self.time_left // 3600, 
    #                                            (self.time_left % 3600) // 60, 
    #                                             self.time_left % 60)
  
    #     self.lcdNumber1.display(time_format1)
    #     self.lcdNumber2.display(time_format2)
    #     QApplication.processEvents()
        # time_used = 1592
        # time_format1 = "{:0>2d}:{:0>2d}:{:0>2d}".format(time_used // 3600, 
        #                                        (time_used % 3600) // 60, 
        #                                         time_used % 60)  
        # time_format1
        
    def open_dataset_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Load the folder with images for identification', self.dataset_path)
        if file_path == None:
            self.label_status.setText('Error','No images path selected')
        else:
            self.dataset_path = file_path
            if not check_path(self.dataset_path):
                self.label_status.setText(u'文件夹名或图像文件名中具有中文字符，请改成纯英文名称！')
                self.dataset_path = ''
                self.comboBox_dataset.setCurrentText('')
            else:
                self.dataset_paths.append(file_path)
                self.comboBox_dataset.addItem(file_path.split("/")[-1])
                self.comboBox_dataset.setCurrentText(file_path.split("/")[-1])
                self.label_status.setText("Image path: %s"%file_path)   

    def renew_dataset_folder(self):
        idx = self.comboBox_dataset.currentIndex()
        print('1809, idx:', idx)
        print(self.dataset_path)
        print(self.dataset_paths)
        self.dataset_path = self.dataset_paths[idx]
        self.label_status.setText("Image path: %s"%self.dataset_path)   


    def open_output_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load Output Folder", self.output_path)
        if file_path == None:
            self.label_status.setText('Error','No save path selected')
        else:
            print(file_path)
            if not check_path(file_path):
                self.label_status.setText(u'文件夹名中具有中文字符，请改成纯英文名称！')
                # self.output_path = ''
                # self.comboBox_output.setCurrentText('')
            else:
                self.output_path = file_path
                self.output_paths.append(file_path)
                self.comboBox_output.addItem(file_path.split("/")[-1])
                self.comboBox_output.setCurrentText(file_path.split("/")[-1])
                self.label_status.setText("Output path: %s"%file_path)   

    def renew_output_folder(self):

        idx = self.comboBox_output.currentIndex()
        self.output_path = self.output_paths[idx]

    def open_export_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load Images Export Folder", self.export_path)
        if file_path == None:
            self.label_status.setText('Error','No save path selected')
        else:
            
            if not check_path(file_path):
                self.label_status.setText(u'文件夹名中具有中文字符，请改成纯英文名称！')
                # self.export_path = ''
                # self.comboBox_export.setCurrentText('')
            else:
                self.export_path = file_path
                self.export_paths.append(file_path)
                self.comboBox_export.addItem(file_path.split("/")[-1])
                self.comboBox_export.setCurrentText(file_path.split("/")[-1])
                self.label_status.setText("Image Export Path: %s"%file_path)   

    def renew_export_folder(self):

        idx = self.comboBox_export.currentIndex()
        self.export_path = self.export_paths[idx]

    def load_formula(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load formula file as *.json', r'./test',
                                               'formula files(*.json)')
        file_path = fname
        if file_path == None:
            self.label_status.setText('Error','No file selected')
        else:
            self.formula_path = file_path
            self.formula_paths.append(file_path)
            self.comboBox_formula.addItem(file_path.split("/")[-1])
            self.comboBox_formula.setCurrentText(file_path.split("/")[-1])
    
    def renew_formula(self):

        idx = self.comboBox_formula.currentIndex()
        self.formula_path = self.formula_paths[idx]    

    def on_mm_yesno(self,code = 0):
        # self.mm_yesno = code
        if code == 0:
            self.mm_yesno = True
            self.groupBox_36.setEnabled(True)
            self.groupBox_42.setEnabled(True)
            self.groupBox_43.setEnabled(True)
            self.label_status.setText("Set Morphology Measurement = %s"%str(self.mm_yesno))
        else:
            self.mm_yesno = False
            self.groupBox_36.setEnabled(False)
            self.groupBox_42.setEnabled(False)
            self.groupBox_43.setEnabled(False)
            self.label_status.setText("Set Morphology Measurement = %s"%str(self.mm_yesno))  

    def size_changed(self):
        try:
            self.pxsize = float(self.lineedit_size.text())
            self.label_status.setText("set up pixel size as %.3f micrometer per pixel"%(self.pxsize))
        except:
            self.label_status.setText("Invalid input for pixel size")

    def on_imagetype(self, code = 0):
        # QRadioButton.toggled()会在 checked=True/checked=False 两种情况下都触发。
        # 这里仅当当前发送者被选中时才更新，避免互斥切换时被“取消选中”的事件覆盖。
        sender = self.sender()
        try:
            if sender is not None and hasattr(sender, "isChecked") and not sender.isChecked():
                return
        except Exception:
            pass
        self.image_type = code
        self.label_status.setText("image type = %s " %str(self.image_types[code]))


    def on_mtype(self):
        combobox = self.sender()
        self.modeltype = self.modeltypes[combobox.currentIndex()]
        self.label_status.setText("Model type: %s selected" %self.modeltype)
        self.update_class()
        self.update_parameter()
        self.save_config()

    def on_device_type(self):
        radiobutton = self.sender()
        if radiobutton.isChecked():
            self.label_status.setText(radiobutton.objectName())
            if radiobutton.objectName() == "flowcam":
                self.device_type = 0
                # self.pushButton_flowcam.setEnabled(True)
                # self.pushButton_microscope.setEnabled(False)
            elif radiobutton.objectName() == "microscope":
                self.device_type = 1
            else:
                self.device_type = 2
                # self.pushButton_flowcam.setEnabled(True)
                # self.pushButton_microscope.setEnabled(False)
    
    def update_parameter(self):
        path_model = r"./Models/%s/"%self.modeltype
        try:
            f = open(os.path.join(path_model, "setting.log"), "r", encoding="utf-8")
            for line in f:
                if "device_type:" in line:
                    self.device_type = int(line.split(":")[1])
                    if self.device_type == 0:
                        self.radioButton_flowcam.setChecked(True)
                    elif self.device_type == 1:
                        self.radioButton_microscope.setChecked(True)
                    elif self.device_type == 2:
                        self.radioButton_ipp.setChecked(True)
        except Exception as e:
            self.label_status.setText(f"error with parameter update: {e}")
                
    def on_mm_select(self, code = 1):
        cbox = self.sender()
        if cbox.isChecked():
            self.mms[code-1] = 1
            # self.label_status.setText('check')
        else:
            self.mms[code-1] = 0
            # print('uncheck')
            
        # mm_list = ["Length","Width","Eq. Diameter","Area","Perimeter",
        #            "Length-ellipse fit","Width-ellipse fit","Area-ellipse fit", 
        #            "Biovolume", "Cell numbers", "Total Volume(multi-celluar)"]
        self.mm_select = []
        for i in range(len(self.mms)):
            if self.mms[i] == 1:
                self.mm_select.append(self.mm_list[i])
        self.label_status.setText("".join(str(self.mm_select)) + "are checked")
          

    def on_level(self):
        radiobutton = self.sender()
        # self.label_status.setText(radiobutton.objectName())
        if radiobutton.isChecked():
            if radiobutton.objectName() == "level0":
                self.level = 0
                if self.dataset_path:
                    self.label_status.setText("Work on current directory:%s" %self.dataset_path)
                else:
                    self.label_status.setText("No images path")
            else:
                self.level = 1
                if self.dataset_path:
                    self.label_status.setText("Work on Sub-Directories of %s" %self.dataset_path)
                else:
                    self.label_status.setText("No images path")
                    
    def update_class(self, fname=None):
        if not fname:
            fname = "./Models/%s/classes_names.txt"%self.modeltype
        self.label_status.setText(fname)
        # path = "./Taxa/classes_names_XM.txt"
        self.loaded_class_path = str(fname)
        # self.listview_class.clear()
        df = pd.read_csv(self.loaded_class_path, encoding="utf-8", header = None)
        # df = pd.read_csv(path, encoding="utf-8", header = None)
        self.taxa_num = len(df)
        self.taxa_list = df.iloc[:,1].values.tolist()
        taxa_df = pd.DataFrame(self.taxa_list, columns = ['Taxa Names'])
        # self.taxa_hab = taxa_df["Taxa Names"].values.tolist()
        self.listmodel = QStandardItemModel(self.taxa_num,2)
        self.listmodel.setHorizontalHeaderLabels(['Class Name','Counts'])
        
        for i in range(self.taxa_num):
            item=QStandardItem(self.taxa_list[i])
            
            self.listmodel.setItem(i,0,item)
        self.listview_class.setModel(self.listmodel)
        self.listview_class.setColumnWidth(0, 150)
        
        self.label_status.setText("Class name list loaded: %s"%self.loaded_class_path)
        self.label_class_num.setText("Taxa number: %d"%self.taxa_num)

    def update_count(self, preds_all, title):
        count = []
        # print(self.features)
        for i in range(self.features):
            count.append(preds_all.count(i))
        # self.count_accumulate = [self.cout_accumulate[x] + count[x] for x in range(len(self.features))]
        df_count = pd.DataFrame(columns=['Class_Names', 'Count'])
        df_count['Class_Names'] = self.taxa_list
        df_count['Count'] = count#self.count_accumulate
        df_count = df_count[7:].sort_values(by='Count', axis=0, ascending=False)
        # update_plot()
        # x = [i for i in range(len(df_count))]
        genus_nms = [x.split("_")[-1] for x in list(df_count["Class_Names"][0:20])]
        # self.update_plot(genus_nms, df_count["Count"][0:20], title)
        for i in range(self.features):
            # print(count[i])
            item=QStandardItem("%d"%count[i])
            item.setTextAlignment(Qt.AlignHCenter)
            self.listmodel.setItem(i,1,item)
        # print(len(count))
        self.listview_class.setModel(self.listmodel)
        QApplication.processEvents()
        # self.label_status.setText("Class name list loaded: %s"%self.loaded_class_path)
        # self.label_class_num.setText("Taxa number: %d"%self.taxa_num)   
                    
        
    def load_class(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'choose a file with csv format', r'./Models/','csv files(*.txt *.csv)')
        if fname:
            self.update_class(fname)
        
    def on_start_iden(self):
        self.save_config()
        if self.dataset_path == "":            
            self.label_status.setText("Error: no image path")            
            return False

        if self.output_path == "":            
            self.label_status.setText("Error: no output path")
            return False
        self.num_total = 0
        self.num_processed = 0
        path_img = self.dataset_path
        self.start = time.time()
        try:
            f = open(os.path.join(r"./Models/%s/"%self.modeltype, "setting.log"), "r", encoding="utf-8")
            for line in f:
                if "features:" in line:
                    # args.features = int(line.split(":")[1])
                    self.features = int(line.split(":")[1])
                    print('Feature number: %d'%self.features)
        except Exception as e:
            print('loading setting.log error:%s'%str(e))
            
        self.pbar1.show()
        self.pbar2.show()
        self.pbar1.setMinimum(0)
        self.pbar1.setMaximum(len(os.listdir(self.dataset_path)))
        self.pbar2.setMinimum(0)

        for dr in os.listdir(path_img):
            if os.path.isdir(os.path.join(path_img, dr)):
                # print(dr)
                self.num_total += sum([is_img(x) for x in os.listdir(os.path.join(path_img, dr))])
                # print(num_total)
        self.time_used = 0
        self.time_left = int(self.num_total * 25) if torch.cuda.is_available() else int(self.num_total * 120)
        self.groupBox_lcd.show()
        self.update_lcd()
        # self.thread2 = timeGO(time_used = self.time_used,
        #               time_left = self.time_left)
        # self.thread2.timeout_signal.connect(self.update_lcd)
        # self.thread2.start()
        # QApplication.processEvents()
        # self.thread1 = idenCNN(main_win=self)
        # self.thread1.signal.connect(self.callback)
        # time_slot =QTimer(self)
        # time_slot.setInterval(1000)
        # time_slot.start()
        # time_slot.timeout.connect(self.lcd_countdown)
        # time_slot.start(1000)
        # QApplication.processEvents()
        # print("mark2")
        # width = self.plot1.width()/200
        # height = self.plot1.height()/200

        # self.plot1.setScene(self.myGraphyScene)
        
        
        self.thread1 = idenCNN(modeltype = self.modeltype,
                               imgs_path = self.dataset_path,
                               output_path = self.output_path,
                               export = self.export_yesno,
                               export_path = self.export_path,
                               mm = self.mm_yesno,
                               mms = self.mms,
                               mm_select = self.mm_select,
                               pxsize = self.pxsize,
                               device_type = self.device_type,
                               level = self.level,
                               image_type = self.image_type)
        
        self.thread1.message_signal.connect(self.message_callback)
        self.thread1.progress_signal.connect(self.progress_callback)
        self.thread1.data_signal.connect(self.data_callback)
        self.thread1.start()
        
        
        # while True:
        #     if not self.thread1.isRunning:
        #         self.thread2.terminate()
        #         self.thread2.quit()
        #         break
        # if not self.thread1 is None:
            # try:
            #     if self.thread1.isRunning:
            #         self.thread1.terminate()
            #         self.thread1.quit()
            # except:
            #     pass
        # self.thread1.start()

    def message_callback(self, msg):
        if not msg == None:
            self.label_status.setText(msg)

    def progress_callback(self, msg):
        if not msg[0] == None:
            self.pbar2.setMaximum(msg[0])
        if not msg[1] == None:
            self.pbar1.setValue(msg[1]+1)
        if not msg[2] == None:
            self.pbar2.setValue(msg[2])
        if not msg[3] == None:
            self.num_processed = int(msg[3])
            self.time_used = int(time.time() - self.start)
            if self.num_processed > 0:
                self.time_left = int((self.time_used / self.num_processed) * (self.num_total - self.num_processed))
            self.time_left = 0 if self.time_left < 1 else self.time_left
            # print('total / processed: %d / %d'%(self.num_total, self.num_processed))
        # if not msg[4] == None:
        #     self.time_left = int(msg[4])
            self.update_lcd()
        # QApplication.processEvents()
        # try:
        #     if self.thread2.isRunning:
        #         self.thread2.terminate()
        #         self.thread2.quit()
        #         print('terminate timGO thread, and start a new one')
        # except:
        #     print('timGO thread not existed, and start a new one')
        #         # pass
        # self.thread2 = timeGO(time_used = self.time_used,
        #                       time_left = self.time_left)
        # self.thread2.timeout_signal.connect(self.update_lcd)
        # self.thread2.start()
            
    def update_lcd(self):
        # time_format = QTime.currentTime()
        # time_format = time_format.toString("hh:mm:ss")
        # if not msg == None:
        # print(*msg)
        # print(self.time_used, self.time_left)
        # if used and left:
        #     self.time_used = int(used) if int(left) > 1 else self.time_used
        #     self.time_left = int(left) if int(left) > 1 else 0
        if not self.time_left == None:
            time_format1 = "{:02.0f}:{:02.0f}:{:02.0f}".format(self.time_used // 3600, 
                                                   (self.time_used % 3600) // 60, 
                                                    self.time_used % 60)  
            time_format2 = "{:02.0f}:{:02.0f}:{:02.0f}".format(self.time_left // 3600, 
                                                   (self.time_left % 3600) // 60, 
                                                    self.time_left % 60)
      
            self.lcdNumber1.display(time_format1)
            self.lcdNumber2.display(time_format2)
            QApplication.processEvents()
                
    def data_callback(self, msg):
        if not msg == None:
            pass
            # # print('data callback')
            # self.update_count(*msg)

    def on_version(self):
        version_box = QMessageBox(self)
        version_box.setWindowTitle("Version")
        version_info = update_version()
        update_date = get_update_date()
        version_box.setText(f"Sorta \nVersion: {version_info} \nUpdate Date: {update_date}")
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        version_box.setFont(font)
        version_box.show()
        # des_path="./"
        
    def on_help(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setText("sorta \nEmail:brigchen@xmu.edu.cn \ncopyright@brigchen \nXiamen University")
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        help_box.setFont(font)
        help_box.show()
        # des_path="./"
    

    def on_export(self, code = 0):
        # QRadioButton.toggled()会在 checked=True/checked=False 两种情况下都触发。
        # 这里仅当当前发送者被选中时才更新，避免互斥切换时被“取消选中”的事件覆盖。
        sender = self.sender()
        try:
            if sender is not None and hasattr(sender, "isChecked") and not sender.isChecked():
                return
        except Exception:
            pass
        self.export_yesno = code
        if code == 1:
            self.pushButton_export.setEnabled(True)
        if code == 0:
            self.pushButton_export.setEnabled(False)
        
        # self.label_status.setText("image type = %d " %self.image_types[code])
        
#%%   
if __name__ =='__main__':
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    app = QApplication(sys.argv) 
    gui = Ui_sorta()
    gui.setWindowIcon(QIcon('./sorta3.png'))
    gui.showMaximized()
    sys.exit(app.exec_())  
