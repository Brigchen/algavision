# -*- coding: utf-8 -*-

# Child_window implementation generated from reading ui file 'test-b.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt,pyqtSignal,QPoint,QRectF
from PyQt5.QtGui import QIntValidator
import sys
import os
import shutil
#%%
class Ui_Viewer(QWidget):
    def __init__(self, img_id, pix):
        super(Ui_Viewer, self).__init__()
        self.image_id = img_id
        self.pixmap = pix
        self.zoomscale = 1
        self.setupUi()
                # print(img_id)
        self.id_show(self.image_id)
        self.img_show(self.pixmap)

    def setupUi(self):
        self.setObjectName("viewer")
        self.resize(1080, 800)
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(50, 1, 50, 1)
        self.horizontalLayout.setSpacing(35)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_zoomin = QtWidgets.QPushButton(self.frame)
        self.pushButton_zoomin.setObjectName("zoom_in")
        self.pushButton_zoomin.clicked.connect(self.on_zoomin_clicked)
        self.horizontalLayout.addWidget(self.pushButton_zoomin)
        self.pushButton_nozoom = QtWidgets.QPushButton(self.frame)
        self.pushButton_nozoom.setObjectName("zoom_no")
        self.pushButton_nozoom.clicked.connect(self.on_zoomno_clicked)
        self.horizontalLayout.addWidget(self.pushButton_nozoom)
        self.pushButton_zoomout = QtWidgets.QPushButton(self.frame)
        self.pushButton_zoomout.setObjectName("zomm_out")
        self.pushButton_zoomout.clicked.connect(self.on_zoomout_clicked)
        self.horizontalLayout.addWidget(self.pushButton_zoomout)
        self.verticalLayout.addWidget(self.frame)
        self.imgView = QtWidgets.QGraphicsView(self)
        self.imgView.setObjectName("view")
        # self.imgView.setAlignment(QtCore.Qt.AlignCenter)
        # self.setCentralWidget(self.imgView)
        self.verticalLayout.addWidget(self.imgView)
        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(QtCore.QSize(0, 20))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("img_name")
        self.verticalLayout.addWidget(self.label)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("viewer", "Image Viewer"))
        self.pushButton_zoomin.setText(_translate("viewer", "Zoom in(+)"))
        self.pushButton_nozoom.setText(_translate("viewer", "Original"))
        self.pushButton_zoomout.setText(_translate("viewer", "Zoom out(-)"))
        self.label.setText(_translate("viewer", "Filename"))

    def id_show(self, image_id):
        self.label.setText(image_id)
    
    def img_show(self,pix):
        self.rect=QRectF(-150,-200,1000,700)
        self.scene=QGraphicsScene()     
        self.imgView.setSceneRect(self.rect)  
        self.item=QGraphicsPixmapItem(pix)                      #创建像素图元
        self.item.setFlags(QGraphicsItem.ItemIsSelectable|QGraphicsItem.ItemIsFocusable|QGraphicsItem.ItemIsMovable)
        self.item.setTransformOriginPoint(500,350)
        # self.item.setPos(QPointF(-400,-300))

        self.scene.addItem(self.item)
        self.imgView.setScene(self.scene)  

    def on_zoomout_clicked(self):

        self.zoomscale=self.zoomscale-0.1
        if self.zoomscale<=0:
            self.zoomscale=0.2
        self.item.setScale(self.zoomscale)   
        self.rect=QRectF(-150,-200,1000,700)
        # self.scene=QGraphicsScene()     
        self.imgView.setSceneRect(self.rect)                               #缩小图像
        # self.set_center() 

    def on_zoomin_clicked(self):

        self.zoomscale=self.zoomscale+0.2
        if self.zoomscale>=2:
            self.zoomscale=2
        self.item.setScale(self.zoomscale)   
        self.rect=QRectF(-150,-200,1000,700)
        # self.scene=QGraphicsScene()     
        self.imgView.setSceneRect(self.rect)  
        # self.set_center() 

    def on_zoomno_clicked(self):
        self.zoomscale = 1
        self.item.setScale(self.zoomscale)
        self.rect=QRectF(-150,-200,1000,700)
        # self.scene=QGraphicsScene()     
        self.imgView.setSceneRect(self.rect)  
        # self.set_center()    

    def set_center(self):
        self.imgView.setAlignment(Qt.AlignCenter)
            
#%%
class Ui_Makedataset(QWidget):
    def __init__(self, class_path = ""):
        super(Ui_Makedataset, self).__init__()
        self.des_paths = []
        self.des_path = ""
        self.txt_paths = []
        self.txt_path = ""
        self.levels = ["Class", "Genus", "Species"]
        self.level = 0
        self.class_path = class_path
        self.ratio = 2
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Dataset")
        self.resize(800, 300)
        # self.setMinimumSize(QtCore.QSize(600, 300))
        # self.setMaximumSize(QtCore.QSize(600, 300))
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_4.addWidget(self.comboBox)
        
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setMaximumSize(QtCore.QSize(150, 3000))
        self.pushButton.setObjectName("pushButton_path")
        self.pushButton.clicked.connect(self.on_dataset_path)
        
        self.horizontalLayout_4.addWidget(self.pushButton)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox()
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_3.addWidget(self.comboBox_2)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setMaximumSize(QtCore.QSize(150, 3000))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(self.on_txt_path)
                
        self.verticalLayout.addWidget(self.groupBox_3)
        
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_2.addWidget(self.radioButton)
        self.radioButton.setChecked(True)
        self.radioButton.toggled.connect(lambda: self.on_level_select(0))
        
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout_2.addWidget(self.radioButton_2)
        self.radioButton_2.toggled.connect(lambda: self.on_level_select(1))
        
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_2.addWidget(self.radioButton_3)
        self.radioButton_3.toggled.connect(lambda: self.on_level_select(2))
        
        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_4 = QtWidgets.QGroupBox(self)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_ratio = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.verticalLayout.addWidget(self.groupBox_4)        
        self.label_ratio = QLabel(self)
        self.label_ratio.setObjectName("Label_ratio")
        self.label_ratio.setText("Ratio of Train set / Test set (Format: Integar, 1~10)")
        self.horizontalLayout_ratio.addWidget(self.label_ratio)
        
        self.lineedit = QLineEdit(self)
        self.lineedit.setObjectName("Edit_ratio")
        self.horizontalLayout_ratio.addWidget(self.lineedit)
        
        pIntvalidator=QIntValidator(self)
        pIntvalidator.setRange(1,10)
        self.lineedit.setValidator(pIntvalidator)
        self.lineedit.editingFinished.connect(self.ratio_change)
        self.lineedit.setText(str(2))
        self.lineedit.setAlignment(Qt.AlignCenter)
        
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(15, -1, 15, -1)
        self.horizontalLayout.setSpacing(14)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_3.clicked.connect(self.on_ok)
        
        self.pushButton_4 = QtWidgets.QPushButton(self.frame)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton_4.clicked.connect(self.on_reset)
        
        self.pushButton_5 = QtWidgets.QPushButton(self.frame)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout.addWidget(self.pushButton_5)
        self.pushButton_5.clicked.connect(self.on_cancel)
        self.verticalLayout.addWidget(self.frame)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint|Qt.WindowCloseButtonHint)
        # self.effect_shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        # self.effect_shadow.setOffset(0,0) # 偏移
        # self.effect_shadow.setBlurRadius(5) # 阴影半径
        # self.effect_shadow.setColor(QtCore.Qt.white) # 阴影颜色
        # self.setGraphicsEffect(self.effect_shadow) 
        color_bg="background-color:#EFEFEF"
        self.setStyleSheet(color_bg)
        # self.setStyleSheet("QWidget#self{border: 2px solid #020202; border-radius: 5px;}")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dataset", "Make Train/Test DataSets"))
        self.label.setText(_translate("Dataset", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Hint:</span> to produce datasets(train set and test set) in <span style=\" font-weight:600; text-decoration: underline;\">three levels</span>, which were determined by the tree structure of class folder.</p></body></html>"))
        self.groupBox_2.setTitle(_translate("Dataset", "Destination Folder"))
        self.pushButton.setText(_translate("Dataset", "Dataset Folder"))
        self.groupBox_3.setTitle(_translate("Dataset", "Txt Folder"))
        self.pushButton_2.setText(_translate("Dataset", "Txt Folder"))
        self.groupBox.setTitle(_translate("Dataset", "Dataset Level"))
        self.radioButton.setText(_translate("Dataset", "Class Level"))
        self.radioButton_2.setText(_translate("Dataset", "Genus Level"))
        self.radioButton_3.setText(_translate("Dataset", "Species Level"))
        self.pushButton_3.setText(_translate("Dataset", "Execute"))
        self.pushButton_4.setText(_translate("Dataset", "Reset"))
        self.pushButton_5.setText(_translate("Dataset", "Exit"))
        self.groupBox_4.setTitle(_translate("Dataset", "Train/Test Ratio"))
        
    def on_dataset_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Load the folder for dataset', './')
        if path:
            self.des_path = path
            self.des_paths.append(path)
            self.comboBox.addItem(str(path))
            self.comboBox.setCurrentText(str(path))
        else:
            QMessageBox.information(self,'Error','No path selected')

    def on_txt_path(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load the folder for classes name txt", "./")
        if file_path:
            self.txt_path = file_path
            self.txt_paths.append(file_path)
            self.comboBox_2.addItem(str(file_path))
            self.comboBox_2.setCurrentText(str(file_path))
        else:
            QMessageBox.information(self,'Error','No filename path selected')
            
    def on_level_select(self, code = 0):
        # levels = ["Class","Genus","Species"]
        if code in [0,1,2]:
            self.level = code
        else:
            self.level = 0
        print(self.levels[self.level])
#%%
    def on_ok(self):
        imgtypes=["png","jpg","tif","bmp","jfif"]
        
        if self.des_path and self.txt_path:
            if self.class_path:
                # QMessageBox.information(self,'Warning!',"Any file in this folder will be removed")
                select = QMessageBox.warning(self,'Warning!',"Any file in this folder will be removed!\n%s \nAre you sure to proceed?"%self.des_path,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
                if select == QMessageBox.No:
                    pass
                else:
                    shutil.rmtree(self.des_path)
                    os.mkdir(self.des_path)
    #%%
                    if self.level == 0:
                        i = 0
                        file = open(self.txt_path +"/class_names.txt", "w")
                        for dr0 in os.listdir(self.class_path):
                            path1 = self.class_path +"/" + dr0
                            if os.path.isdir(path1):                                
                                fl_list = []
                                fl_list = self.get_filelist(path1, fl_list)
                                if len(fl_list) > 0:                                
                                    self.label.setText("Starting make datasets, don't close this window\n\nTaxa: %s, images counts: %d"%(dr0, len(fl_list)))
                                    QApplication.processEvents()

                                    file.write("%d, %s\n"%(i,dr0))
                                    des_train_path = os.path.join(self.des_path, "train/%d/"%i)
                                    des_test_path = os.path.join(self.des_path, "test/%d/"%i)
                                    i += 1
                                    
                                    if not os.path.exists(des_train_path):
                                        os.makedirs(des_train_path)
                                    if not os.path.exists(des_test_path):
                                        os.makedirs(des_test_path)
                                    for j, fl in enumerate(fl_list):
                                        if fl.split(".")[-1] in imgtypes:
                                            if j%(self.ratio + 1) == 0:
                                                try: 
                                                    shutil.copy(fl, des_test_path)
                                                except Exception as e:
                                                    print(e)
                                            else:
                                                try: 
                                                    shutil.copy(fl, des_train_path)
                                                except Exception as e:
                                                    print(e)
                                
                        file.close()
    # for a, b, c in os.walk(path):
    #     print(c)
    #%%
                    if self.level == 1:
                        i = 0
                        file = open(self.txt_path +"/class_names.txt", "w")
                        for dr0 in os.listdir(self.class_path):
                            path1 = self.class_path +"/" + dr0
                            if os.path.isdir(path1):
                                for dr1 in os.listdir(path1):
                                    path2 = path1 +"/" + dr1
                                    if os.path.isdir(path2):
                                        fl_list = []
                                        fl_list = self.get_filelist(path2, fl_list)
                                        if len(fl_list) > 0:                                           
                                            self.label.setText("Starting make datasets, don't close this window\n\nTaxa: %s, images counts: %d"%(dr0+"__"+dr1, len(fl_list)))                                        
                                            file.write("%d, %s\n"%(i,"__".join([dr0, dr1])))
                                            QApplication.processEvents()                                     
                                            des_train_path = os.path.join(self.des_path, "train/%d/"%i)
                                            des_test_path = os.path.join(self.des_path, "test/%d/"%i)
                                            i += 1
                                            
                                            if not os.path.exists(des_train_path):
                                                os.makedirs(des_train_path)
                                            if not os.path.exists(des_test_path):
                                                os.makedirs(des_test_path)
                                            for j, fl in enumerate(fl_list):
                                                if fl.split(".")[-1] in imgtypes:
                                                    if j%(self.ratio + 1) == 0:
                                                        try: 
                                                            shutil.copy(fl, des_test_path)
                                                        except Exception as e:
                                                            print(e)
                                                    else:
                                                        try: 
                                                            shutil.copy(fl, des_train_path)
                                                        except Exception as e:
                                                            print(e)
                        file.close()
    #%%
                    if self.level == 2:
                        i = 0
                        file = open(self.txt_path +"/class_names.txt", "w")
                        for dr0 in os.listdir(self.class_path):
                            path1 = self.class_path +"/" + dr0
                            if os.path.isdir(path1):
                                for dr1 in os.listdir(path1):
                                    path2 = path1 +"/" + dr1
                                    if os.path.isdir(path2):
                                        for dr2 in os.listdir(path2):
                                            path3 = path2 + "/" +dr2
                                            if os.path.isdir(path3):
                                                fl_list = []
                                                fl_list = self.get_filelist(path3, fl_list)
                                                if len(fl_list) > 0:                                                  
                                                    self.label.setText("Starting make datasets, don't close this window\n\nTaxa: %s, images counts: %d"%(dr0+"__"+dr1+"__"+dr2, len(fl_list)))                                        
                                                    QApplication.processEvents()
                                                    file.write("%d, %s\n"%(i,"__".join([dr0, dr1, dr2])))
                                                    des_train_path = os.path.join(self.des_path, "train/%d/"%i)
                                                    des_test_path = os.path.join(self.des_path, "test/%d/"%i)
                                                    i += 1                                            
                                                    if not os.path.exists(des_train_path):
                                                        os.makedirs(des_train_path)
                                                    if not os.path.exists(des_test_path):
                                                        os.makedirs(des_test_path)
                                                    for j, fl in enumerate(fl_list):
                                                        if fl.split(".")[-1] in imgtypes:
                                                            if j%(self.ratio + 1) == 0:
                                                                try: 
                                                                    shutil.copy(fl, des_test_path)
                                                                except Exception as e:
                                                                    print(e)
                                                            else:
                                                                try: 
                                                                    shutil.copy(fl, des_train_path)
                                                                except Exception as e:
                                                                    print(e)
                        file.close()
    #%%
                    file_train = open(self.txt_path + "/train.txt", "w")
                    file_test = open(self.txt_path + "/test.txt", "w")
                    path_train = self.des_path+"/train/"
                    path_test = self.des_path+"/test/"
                    for dr in os.listdir(path_train):                        
                        i = 0
                        if os.path.isdir(path_train + dr +"/"):
                            path_dr = path_train + dr +"/"
                            for fl in os.listdir(path_dr):
                                path_fl = path_dr + fl
                                if os.path.isfile(path_fl):
                                    file_train.write("%d,%s\n"%(i,self.des_path+"/train/" + dr +"/" + fl))
                                    i += 1
                    file_train.close()
                    # self.label.setText("Starting make datasets, don't close this window\n\nTaxa: %s, images counts: %d"%(dr0, len(fl_list)))                                        

                    for dr in os.listdir(path_test):                        
                        i = 0
                        if os.path.isdir(path_train + dr +"/"):
                            path_dr = path_train + dr +"/"
                            for fl in os.listdir(path_dr):
                                path_fl = path_dr + fl
                                if os.path.isfile(path_fl):
                                    file_test.write("%d,%s\n"%(i,self.des_path+"/test/" + dr +"/" + fl))
                                    i += 1
                    file_test.close()
                    QApplication.processEvents()
                    self.label.setText("Finish make datasets!\nCheck the train/test datasets in :\n%s"%self.des_path)                                       

                    
            else:
                QMessageBox.information(self,'Error','Class folder not determined!')
        else:
            QMessageBox.information(self,'Error','Setup before execution!')
        # self.close()
#%%
    def ratio_change(self):
        self.ratio = int(self.lineedit.text())
        print(self.ratio)
#%%    funny: 递归函数
    def get_filelist(self, fl, Filelist):
        imgtypes=["png","jpg","tif","bmp","jfif"]    
        if os.path.isfile(fl) and (fl.split(".")[-1] in imgtypes):
            Filelist.append(fl)                
        else:
            if os.path.isdir(fl):
                for s in os.listdir(fl):
                    newDir=os.path.join(fl,s)
                    self.get_filelist(newDir, Filelist)
            
        return Filelist


#%%
    def on_cancel(self):
        self.close()
        
    def on_reset(self):
        self.show()


#%%
class Ui_Enhance(QWidget):
    def __init__(self, dataset_path = ""):
        super(Ui_Enhance, self).__init__()
        self.des_paths = []
        self.des_path = ""
        self.txt_paths = []
        self.txt_path = ""
        self.etypes = ["HFlip", "VFlip", "RCont", "RBrig", "RRota", "RDefo", "RZoom", "RSatu"]
        self.etypes_select = []
        self.dataset_path = dataset_path
        self.test_no = 500
        self.train_no = 1000
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Enhance")
        self.resize(800, 500)
        # self.setMinimumSize(QtCore.QSize(600, 300))
        # self.setMaximumSize(QtCore.QSize(600, 300))
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        
        self.groupBox_dataset = QtWidgets.QGroupBox(self)
        self.groupBox_dataset.setObjectName("groupBox_dataset")
        self.groupBox_dataset.setTitle("Datasets Directory")
        self.verticalLayout.addWidget(self.groupBox_dataset)
        self.verticalLayout_dataset = QtWidgets.QVBoxLayout(self.groupBox_dataset)
        self.frame_dataset1 = QtWidgets.QFrame(self.groupBox_dataset)
        self.frame_dataset2 = QtWidgets.QFrame(self.groupBox_dataset)
        self.verticalLayout_dataset.addWidget(self.frame_dataset1)
        self.verticalLayout_dataset.addWidget(self.frame_dataset2)
        self.horizontalLayout_dataset1 = QtWidgets.QHBoxLayout(self.frame_dataset1)
        self.radioButton_dataset1 = QtWidgets.QRadioButton(self.frame_dataset1)
        self.radioButton_dataset1.setText("Original Directory")
        self.radioButton_dataset1.setChecked(True)
        self.radioButton_dataset2 = QtWidgets.QRadioButton(self.frame_dataset1)
        self.radioButton_dataset2.setText("New Directory")
        self.horizontalLayout_dataset1.addWidget(self.radioButton_dataset1)
        self.horizontalLayout_dataset1.addWidget(self.radioButton_dataset2)        
        
        self.horizontalLayout_dataset2 = QtWidgets.QHBoxLayout(self.frame_dataset2)
        self.comboBox = QtWidgets.QComboBox(self.frame_dataset2)
        self.horizontalLayout_dataset2.addWidget(self.comboBox)
        
        self.pushButton = QtWidgets.QPushButton(self.frame_dataset2)
        self.pushButton.setMaximumSize(QtCore.QSize(150, 3000))
        self.pushButton.setText("New Dataset Directory")
        self.pushButton.clicked.connect(self.on_dataset_path)
        self.horizontalLayout_dataset2.addWidget(self.pushButton)

        self.groupBox_txt = QtWidgets.QGroupBox()
        self.groupBox_txt.setTitle("Directory for List File")
        self.verticalLayout.addWidget(self.groupBox_txt)
        
        self.horizontalLayout_txt = QtWidgets.QHBoxLayout(self.groupBox_txt)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_txt)
        self.horizontalLayout_txt.addWidget(self.comboBox_2)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_txt)
        self.pushButton_2.setMaximumSize(QtCore.QSize(150, 3000))
        self.pushButton_2.setText("Open Output Directory")
        self.horizontalLayout_txt.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(self.on_txt_path)
                       
        self.groupBox_etypes = QtWidgets.QGroupBox(self)
        self.groupBox_etypes.setTitle("Image Transform Types")
        self.verticalLayout.addWidget(self.groupBox_etypes)
        self.gridLayout_etypes = QtWidgets.QGridLayout(self.groupBox_etypes)
        self.checkbox_1 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_1.setText("Horizonal flip")
        self.checkbox_1.setChecked(False)
        self.checkbox_1.stateChanged.connect(lambda: self.on_etypes_select(1))
        self.checkbox_2 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_2.setText("Vertical flip")
        self.checkbox_2.setChecked(False)
        self.checkbox_2.stateChanged.connect(lambda: self.on_etypes_select(2))
        self.checkbox_3 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_3.setText("Random contrast")
        self.checkbox_3.setChecked(False)
        self.checkbox_3.stateChanged.connect(lambda: self.on_etypes_select(3))
        self.checkbox_4 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_4.setText("Random brighten")
        self.checkbox_4.setChecked(False)
        self.checkbox_4.stateChanged.connect(lambda: self.on_etypes_select(4))
        self.checkbox_5 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_5.setText("Random rotation")
        self.checkbox_5.setChecked(False)
        self.checkbox_5.stateChanged.connect(lambda: self.on_etypes_select(5))
        self.checkbox_6 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_6.setText("Random deform")
        self.checkbox_6.setChecked(False)
        self.checkbox_6.stateChanged.connect(lambda: self.on_etypes_select(6))
        self.checkbox_7 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_7.setText("Random zoom")
        self.checkbox_7.setChecked(False)
        self.checkbox_7.stateChanged.connect(lambda: self.on_etypes_select(7))
        self.checkbox_8 = QtWidgets.QCheckBox(self.groupBox_etypes)
        self.checkbox_8.setText("Random saturation")
        self.checkbox_8.setChecked(False)
        self.checkbox_8.stateChanged.connect(lambda: self.on_etypes_select(8))
        
        self.gridLayout_etypes.addWidget(self.checkbox_1, 0, 0, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_2, 1, 0, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_3, 2, 0, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_4, 3, 0, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_5, 0, 1, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_6, 1, 1, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_7, 2, 1, 1, 1)
        self.gridLayout_etypes.addWidget(self.checkbox_8, 3, 1, 1, 1)
        
        
        self.groupBox_num = QtWidgets.QGroupBox(self)
        self.groupBox_num.setTitle("Minimum Images of Datasets")
        self.verticalLayout.addWidget(self.groupBox_num)   
        self.verticalLayout_num = QtWidgets.QVBoxLayout(self.groupBox_num)
        self.frame_num1 = QtWidgets.QFrame(self.groupBox_num)
        self.frame_num2 = QtWidgets.QFrame(self.groupBox_num)
        self.verticalLayout_num.addWidget(self.frame_num1)
        self.verticalLayout_num.addWidget(self.frame_num2)
        self.horizontalLayout_num1 = QtWidgets.QHBoxLayout(self.frame_num1)
        self.label_num_train = QLabel(self)
        self.label_num_train.setText("Set Minimum Images in a class of Train set")
        self.lineedit_train = QLineEdit(self)
        self.horizontalLayout_num1.addWidget(self.lineedit_train)       
        pIntvalidator=QIntValidator(self)
        pIntvalidator.setRange(1,100000)
        self.lineedit_train.setValidator(pIntvalidator)
        # self.lineedit_train.editingFinished.connect(self.num_train_change)
        self.lineedit_train.setText(str(1000))
        self.lineedit_train.setAlignment(Qt.AlignCenter)        
        
        self.horizontalLayout_num1.addWidget(self.label_num_train)
        self.horizontalLayout_num1.addWidget(self.lineedit_train)        
        
        self.horizontalLayout_num2 = QtWidgets.QHBoxLayout(self.frame_num2)
        self.label_num_test = QLabel(self)
        self.label_num_test.setText("Set Minimum Images in a class of Test set")
        self.lineedit_test = QLineEdit(self)
        self.horizontalLayout_num2.addWidget(self.lineedit_test)       
        pIntvalidator=QIntValidator(self)
        pIntvalidator.setRange(1,100000)
        self.lineedit_test.setValidator(pIntvalidator)
        # self.lineedit_test.editingFinished.connect(self.num_test_change)
        self.lineedit_test.setText(str(500))
        self.lineedit_test.setAlignment(Qt.AlignCenter)      
        self.horizontalLayout_num2.addWidget(self.label_num_test)
        self.horizontalLayout_num2.addWidget(self.lineedit_test)
        
        
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(15, -1, 15, -1)
        self.horizontalLayout.setSpacing(14)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.pushButton_ok = QtWidgets.QPushButton(self.frame)
        self.pushButton_ok.setText("Excecute")
        self.horizontalLayout.addWidget(self.pushButton_ok)
        self.pushButton_ok.clicked.connect(self.on_ok)
        
        self.pushButton_reset = QtWidgets.QPushButton(self.frame)
        self.pushButton_reset.setText("Reset")
        self.horizontalLayout.addWidget(self.pushButton_reset)
        self.pushButton_reset.clicked.connect(self.on_reset)
        
        self.pushButton_cancel = QtWidgets.QPushButton(self.frame)
        self.pushButton_cancel.setText("Cancel")
        self.horizontalLayout.addWidget(self.pushButton_cancel)
        self.pushButton_cancel.clicked.connect(self.on_cancel)
        self.verticalLayout.addWidget(self.frame)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint|Qt.WindowCloseButtonHint)
        # self.effect_shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        # self.effect_shadow.setOffset(0,0) # 偏移
        # self.effect_shadow.setBlurRadius(5) # 阴影半径
        # self.effect_shadow.setColor(QtCore.Qt.white) # 阴影颜色
        # self.setGraphicsEffect(self.effect_shadow) 
        color_bg="background-color:#EFEFEF"
        self.setStyleSheet(color_bg)
        # self.setStyleSheet("QWidget#self{border: 2px solid #020202; border-radius: 5px;}")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Enhance", "Train/Test DataSets Enhancement"))
        self.label.setText(_translate("Enhance", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Hint:</span> to produce datasets(train set and test set) in <span style=\" font-weight:600; text-decoration: underline;\">three levels</span>, which were determined by the tree structure of class folder.</p></body></html>"))
        # self.groupBox_dataset.setTitle(_translate("Dataset", "Destination Folder"))
        # self.pushButton.setText(_translate("Dataset", "Dataset Folder"))
        # self.groupBox_txt.setTitle(_translate("Dataset", "Txt Folder"))
        # self.pushButton_2.setText(_translate("Dataset", "Txt Folder"))
        # self.groupBox.setTitle(_translate("Dataset", "Dataset Level"))
        # self.radioButton.setText(_translate("Dataset", "Class Level"))
        # self.radioButton_2.setText(_translate("Dataset", "Genus Level"))
        # self.radioButton_3.setText(_translate("Dataset", "Species Level"))
        # self.pushButton_3.setText(_translate("Dataset", "Execute"))
        # self.pushButton_4.setText(_translate("Dataset", "Reset"))
        # self.pushButton_5.setText(_translate("Dataset", "Exit"))
        # self.groupBox_4.setTitle(_translate("Dataset", "Train/Test Ratio"))
        
    def on_dataset_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Load a folder for enhanced dataset', './')
        if path:
            self.des_path = path
            self.des_paths.append(path)
            self.comboBox.addItem(str(path))
            self.comboBox.setCurrentText(str(path))
        else:
            QMessageBox.information(self,'Error','No path selected')

    def on_txt_path(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load a folder for output of the image list", "./")
        if file_path:
            self.txt_path = file_path
            self.txt_paths.append(file_path)
            self.comboBox_2.addItem(str(file_path))
            self.comboBox_2.setCurrentText(str(file_path))
        else:
            QMessageBox.information(self,'Error','No filename path selected')
            
    def on_etypes_select(self, code = 1):
        cbox = self.sender()
        if cbox.isChecked:
            self.etypes_select.append(self.etypes[code])
        else:
            self.etypes_select.drop(self.etypes[code])
        print(self.etypes_select, "are checked")
#%%
    def on_ok(self):
        imgtypes=["png","jpg","tif","bmp","jfif"]
        
        if self.des_path and self.txt_path:
            if self.dataset_path:
                # QMessageBox.information(self,'Warning!',"Any file in this folder will be removed")
                select = QMessageBox.warning(self,'Warning!',"Any file in this folder will be removed!\n%s \nAre you sure to proceed?"%self.des_path,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
                if select == QMessageBox.No:
                    pass
                else:
                    shutil.rmtree(self.des_path)
                    os.mkdir(self.des_path)
   
    #%%
                    file_train = open(self.txt_path + "/train.txt", "w")
                    file_test = open(self.txt_path + "/test.txt", "w")
                    path_train = self.des_path+"/train/"
                    path_test = self.des_path+"/test/"
                    for dr in os.listdir(path_train):                        
                        i = 0
                        if os.path.isdir(path_train + dr +"/"):
                            path_dr = path_train + dr +"/"
                            for fl in os.listdir(path_dr):
                                path_fl = path_dr + fl
                                if os.path.isfile(path_fl):
                                    file_train.write("%d,%s\n"%(i,self.des_path+"/train/" + dr +"/" + fl))
                                    i += 1
                    file_train.close()
                    # self.label.setText("Starting make datasets, don't close this window\n\nTaxa: %s, images counts: %d"%(dr0, len(fl_list)))                                        

                    for dr in os.listdir(path_test):                        
                        i = 0
                        if os.path.isdir(path_train + dr +"/"):
                            path_dr = path_train + dr +"/"
                            for fl in os.listdir(path_dr):
                                path_fl = path_dr + fl
                                if os.path.isfile(path_fl):
                                    file_test.write("%d,%s\n"%(i,self.des_path+"/test/" + dr +"/" + fl))
                                    i += 1
                    file_test.close()
                    QApplication.processEvents()
                    self.label.setText("Finish make datasets!\nCheck the train/test datasets in :\n%s"%self.des_path)                                       

                    
            else:
                QMessageBox.information(self,'Error','Class folder not determined!')
        else:
            QMessageBox.information(self,'Error','Setup before execution!')
        # self.close()
#%%
    def ratio_change(self):
        self.ratio = int(self.lineedit.text())
        print(self.ratio)
#%%    funny: 递归函数
    def get_filelist(self, fl, Filelist):
        imgtypes=["png","jpg","tif","bmp","jfif"]    
        if os.path.isfile(fl) and (fl.split(".")[-1] in imgtypes):
            Filelist.append(fl)                
        else:
            if os.path.isdir(fl):
                for s in os.listdir(fl):
                    newDir=os.path.join(fl,s)
                    self.get_filelist(newDir, Filelist)
            
        return Filelist


#%%
    def on_cancel(self):
        self.close()
        
    def on_reset(self):
        self.show()
#%%
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    image_id = "./Class"
    pix = QPixmap(image_id)
    gui_data = Ui_Enhance(image_id)
    gui_data.setWindowIcon(QIcon('./Modler_icon_small.png'))
    gui_data.show()
    app.exec_()