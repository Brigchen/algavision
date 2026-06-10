# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
# from PyQt5.QtCore import QStringListModel,QAbstractListModel,QModelIndex
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from child import Ui_Viewer, Ui_Makedataset
# from tools import getInput
import shutil
import os
import sys
import cv2
import numpy as np
import imageio
from imagecut import COI_Save, Flowcam_Cut

#%%
class Ui_Segman(QMainWindow):
    def __init__(self):
        super(Ui_Segman, self).__init__()
        self.work_paths = []
        self.work_path = ""
        self.save_paths = []
        self.save_path =""
        self.tree_path = ""
        self.select_ids = []
        self.image_ids = []
        self.image_list = []
        self.displayed_image_size = 300
        self.col = 0
        self.row = 0
        self.num_a_page = 12
        self.size_min = 1
        self.pixelsize = 0.25
        self.imagetype = 0
        self.magnification = 10
        self.connectivity = 1
        self.level = 1
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Segman")
        self.setWindowTitle('Segman: Segmentation of images')
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
        self.label.setPixmap(QPixmap("Segman_logo2.png"))
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
        self.gridLayout_2 = QGridLayout(self.frame_top_right)
        self.gridLayout_2.setContentsMargins(0, 0, 10, 0)
        self.gridLayout_2.setHorizontalSpacing(15)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        self.pushButton_seg = QPushButton(self.frame_top_right)
        self.pushButton_seg.setObjectName("segman")
        self.pushButton_seg.clicked.connect(self.on_start_segman)
        self.gridLayout_2.addWidget(self.pushButton_seg, 0, 0, 1, 1)
        
        self.pushButton_tag = QPushButton(self.frame_top_right)
        self.pushButton_tag.setObjectName("Segman")
        self.pushButton_tag.clicked.connect(self.on_start_tagor)
        self.gridLayout_2.addWidget(self.pushButton_tag, 0, 1, 1, 1)
        
        self.pushButton_mod = QPushButton(self.frame_top_right)
        self.pushButton_mod.setObjectName("modler")
        self.pushButton_mod.clicked.connect(self.on_start_modler)
        self.gridLayout_2.addWidget(self.pushButton_mod, 0, 2, 1, 1)
        
        self.pushButton_sor = QPushButton(self.frame_top_right)
        self.pushButton_sor.setObjectName("sorta")
        self.pushButton_sor.clicked.connect(self.on_start_modler)
        self.gridLayout_2.addWidget(self.pushButton_sor, 0, 3, 1, 1)
        
        self.pushButton_start = QPushButton(self.frame_top_right)
        self.pushButton_start.setObjectName("start")
        self.pushButton_start.clicked.connect(self.on_segmentation)
        self.gridLayout_2.addWidget(self.pushButton_start, 1, 0, 1, 2)    
                       
        self.pushButton_che = QPushButton(self.frame_top_right)
        self.pushButton_che.setObjectName("datasets")
        self.pushButton_che.clicked.connect(self.on_check)
        self.gridLayout_2.addWidget(self.pushButton_che, 1, 2, 1, 2)
        
        self.horizontalLayout.addWidget(self.frame_top_right)
       
        self.gridLayout.addWidget(self.frame_top, 0, 0, 1, 1)
        self.frame_down = QFrame(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down.sizePolicy().hasHeightForWidth())
        self.frame_down.setSizePolicy(sizePolicy)
        self.frame_down.setFrameShape(QFrame.StyledPanel)
        self.frame_down.setFrameShadow(QFrame.Raised)
        self.frame_down.setObjectName("frame_down")
        
        self.horizontalLayout_2 = QHBoxLayout(self.frame_down)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 10)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.frame_down_left = QFrame(self.frame_down)
        self.frame_down_left.setMinimumSize(QSize(250, 0))
        self.frame_down_left.setMaximumSize(QSize(350, 16777215))
        self.frame_down_left.setFrameShape(QFrame.StyledPanel)
        self.frame_down_left.setFrameShadow(QFrame.Raised)
        self.frame_down_left.setObjectName("frame_down_left")
        
        self.verticalLayout = QVBoxLayout(self.frame_down_left)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.groupBox = QGroupBox(self.frame_down_left)
        self.groupBox.setMinimumSize(QSize(0, 100))
        self.groupBox.setObjectName("working")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.pushButton_wor = QPushButton(self.groupBox)
        self.pushButton_wor.setObjectName("working_btn")
        self.verticalLayout_2.addWidget(self.pushButton_wor)
        self.pushButton_wor.clicked.connect(self.open_work_folder)
        
        self.comboBox_wor = QComboBox(self.groupBox)
        self.comboBox_wor.setObjectName("working_cb")
        self.comboBox_wor.currentIndexChanged.connect(self.renew_work_folder) 
        self.verticalLayout_2.addWidget(self.comboBox_wor)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout_2.addWidget(self.frame_down_left)
        
        
        self.groupBox_2 = QGroupBox(self.frame_down_left)
        self.groupBox_2.setMinimumSize(QSize(0, 100))
        self.groupBox_2.setObjectName("class")
        
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_sav = QPushButton(self.groupBox_2)
        self.pushButton_sav.setObjectName("class_btn")
        self.verticalLayout_3.addWidget(self.pushButton_sav)
        self.pushButton_sav.clicked.connect(self.open_save_folder)
        
        self.comboBox_sav = QComboBox(self.groupBox_2)
        self.comboBox_sav.setObjectName("class_cb")
        self.comboBox_sav.currentIndexChanged.connect(self.renew_save_folder) 
        self.verticalLayout_3.addWidget(self.comboBox_sav)

        self.groupBox_3 = QGroupBox(self.frame_down_left)                        #Settings
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())      
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.groupBox_31 = QGroupBox(self.groupBox_3)    #Image types
        self.groupBox_31.setObjectName("Image types")
        self.groupBox_31.setMinimumSize(QSize(200, 50))
        self.groupBox_31.setMaximumSize(QSize(500, 100))
        
        self.verticalLayout_4.addWidget(self.groupBox_31)
        self.radioButton_Microscope = QRadioButton(self.groupBox_31)
        self.radioButton_Microscope.setChecked(True)
        self.radioButton_Microscope.setObjectName("Microscope")
        self.radioButton_Microscope.toggled.connect(self.on_imagetype)
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_31)
        self.verticalLayout_5.addWidget(self.radioButton_Microscope)
        self.radioButton_Flowcam = QRadioButton(self.groupBox_31)
        self.radioButton_Flowcam.setObjectName("Flowcam")
        self.verticalLayout_5.addWidget(self.radioButton_Flowcam)
        self.radioButton_Flowcam.toggled.connect(self.on_imagetype)

        self.groupBox_32 = QGroupBox(self.groupBox_3)    #min size
        self.groupBox_32.setObjectName("Minimum Size")
        self.groupBox_32.setMinimumSize(QSize(200, 75))
        self.groupBox_32.setMaximumSize(QSize(500, 100))
        self.verticalLayout_4.addWidget(self.groupBox_32)
        self.lineEdit_size = QDoubleSpinBox(self.groupBox_32)
        # self.lineEdit_size.setValidator(QtGui.QIntValidator(0,100))
        self.lineEdit_size.setGeometry(QtCore.QRect(30, 30, 130, 30))
        self.lineEdit_size.setProperty("value", 1)
        self.lineEdit_size.setObjectName("size_min")
        self.label_size = QtWidgets.QLabel(self.groupBox_32)
        self.label_size.setGeometry(QtCore.QRect(170, 30, 30, 30))
        self.label_size.setText("um")

        self.groupBox_33 = QGroupBox(self.groupBox_3)    #magnification
        self.groupBox_33.setObjectName("Objective lens magnification")
        self.groupBox_33.setMinimumSize(QSize(200, 75))
        self.groupBox_33.setMaximumSize(QSize(500, 100))
        self.verticalLayout_4.addWidget(self.groupBox_33)
        self.comboBox_mag=QComboBox(self.groupBox_33)
        self.comboBox_mag.addItems(['0.5X','1X','2X', "4X", "10X", "20X", "40X","100X"])
        self.comboBox_mag.setCurrentIndex(4)
        self.comboBox_mag.setGeometry(QtCore.QRect(30, 30, 130, 30))
        
        self.groupBox_34 = QGroupBox(self.groupBox_3)    #resolution
        self.groupBox_34.setObjectName("Pixel size")
        self.groupBox_34.setMinimumSize(QSize(200, 75))
        self.groupBox_34.setMaximumSize(QSize(500, 100))

        self.verticalLayout_4.addWidget(self.groupBox_34)
        self.lineEdit_pixel = QLineEdit(self.groupBox_34)
        self.lineEdit_pixel.setObjectName("pixelsize")
        # self.lineEdit_pixel.setValidator(QtGui.QDoubleValidator(0,10,2))
        self.lineEdit_pixel.setGeometry(QtCore.QRect(20, 30, 100, 30))
        self.lineEdit_pixel.setProperty("value", 0.25)
        self.lineEdit_pixel.setText("0.25")
        self.label_pixel = QLabel(self.groupBox_34)
        self.label_pixel.setGeometry(QtCore.QRect(130, 30, 80, 30))
        self.label_pixel.setText("um / pixel")

        self.groupBox_35 = QGroupBox(self.groupBox_3)    #magnification
        self.groupBox_35.setObjectName("Connectivity(Microscope)")
        self.groupBox_35.setMinimumSize(QSize(200, 100))
        self.groupBox_35.setMaximumSize(QSize(500, 150))
        self.verticalLayout_4.addWidget(self.groupBox_35)
        self.radioButton_High = QRadioButton(self.groupBox_35)
        self.radioButton_High.setObjectName("High")
        self.radioButton_High.setText("High")
        self.radioButton_Mid = QRadioButton(self.groupBox_35)
        # self.radioButton_High.setChecked(True)
        self.radioButton_Mid.setObjectName("Mid")
        self.radioButton_Mid.setText("Middle")
        self.radioButton_Mid.setChecked(True)
        self.radioButton_Low = QRadioButton(self.groupBox_35)
        # self.radioButton_High.setChecked(True)
        self.radioButton_Low.setObjectName("Low")
        self.radioButton_Low.setText("Low")
        self.verticalLayout_sens = QVBoxLayout(self.groupBox_35)
        self.verticalLayout_sens.addWidget(self.radioButton_High)
        self.verticalLayout_sens.addWidget(self.radioButton_Mid)
        self.verticalLayout_sens.addWidget(self.radioButton_Low)
        self.radioButton_High.toggled.connect(self.on_connectivity)
        self.radioButton_Mid.toggled.connect(self.on_connectivity)
        self.radioButton_Low.toggled.connect(self.on_connectivity)
        

        self.groupBox_36 = QGroupBox(self.groupBox_3)    #magnification
        self.groupBox_36.setObjectName("levels")
        # self.groupBox_35.setMinimumSize(QSize(200, 50))
        # self.groupBox_35.setMaximumSize(QSize(500, 100))
        self.verticalLayout_4.addWidget(self.groupBox_36)
        self.groupBox_36.setSizePolicy(sizePolicy)
        self.radioButton_level0 = QRadioButton(self.groupBox_36)
        self.radioButton_level0.setObjectName("level0")
        self.radioButton_level0.setText("Current directory")
        self.radioButton_level1 = QRadioButton(self.groupBox_36)
        self.radioButton_level1.setObjectName("level1")
        self.radioButton_level1.setText("Sub-directories")
        self.radioButton_level1.setChecked(True)
        self.verticalLayout_level = QVBoxLayout(self.groupBox_36)
        self.verticalLayout_level.addWidget(self.radioButton_level0)
        self.verticalLayout_level.addWidget(self.radioButton_level1)
        self.radioButton_level0.toggled.connect(self.on_level)
        self.radioButton_level1.toggled.connect(self.on_level)
        
        self.frame_down_right = QFrame(self.frame_down)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down_right.sizePolicy().hasHeightForWidth())
        self.frame_down_right.setSizePolicy(sizePolicy)
        self.frame_down_right.setFrameShape(QFrame.StyledPanel)
        self.frame_down_right.setFrameShadow(QFrame.Raised)
        self.frame_down_right.setObjectName("frame_6")
        
        self.verticalLayout_5 = QVBoxLayout(self.frame_down_right)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.groupBox_status = QGroupBox(self.frame_down_right)
        self.groupBox_status.setObjectName("status")
        self.verticalLayout_5.addWidget(self.groupBox_status)
        self.groupBox_status.setMinimumHeight(100)
        self.groupBox_status.setMaximumSize(5000,100)
        self.groupBox_status.setTitle("Status")
        # sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.groupBox_status.sizePolicy().hasHeightForWidth())
        # self.groupBox_status.setSizePolicy(sizePolicy)
        
        self.label_status = QLabel(self.groupBox_status)
        self.label_status.setText("Welcome to use Segmen tool")
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_status)
        self.verticalLayout_6.addWidget(self.label_status)
        
        self.label_progress = QLabel(self.groupBox_status)
        # self.label_progress.setText("Welcome to use Segmen tool")
        self.verticalLayout_6.addWidget(self.label_progress)
        # self.progressBar = QtWidgets.QProgressBar(self.groupBox_status)

        self.groupBox_img = QGroupBox(self.frame_down_right)
        self.groupBox_img.setTitle("Image Browser")
        self.verticalLayout_5.addWidget(self.groupBox_img)
        # sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.groupBox_img.sizePolicy().hasHeightForWidth())
        # self.groupBox_img.setSizePolicy(sizePolicy)
        
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox_img)
        self.frame_imgview = QFrame(self.groupBox_img)
        self.horizontalLayout_4.addWidget(self.frame_imgview)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

        self.scrollArea = QScrollArea(self.frame_imgview)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("image_show_area")
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 772, 560))
        self.scrollAreaWidgetContents.setObjectName("image_show")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_images = QGridLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_7 = QVBoxLayout(self.frame_imgview)
        self.verticalLayout_7 .setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.addWidget(self.scrollArea)

        self.frame_8 = QFrame(self.frame_imgview)
        self.frame_8.setMinimumSize(QSize(0, 45))
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_7.addWidget(self.frame_8)
        
        self.horizontalLayout_5 = QHBoxLayout(self.frame_8)
        # self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.pushButton_las = QPushButton(self.frame_8)
        self.pushButton_las.setObjectName("last_btn")
        self.pushButton_las.clicked.connect(self.last_page)
        self.horizontalLayout_5.addWidget(self.pushButton_las)
        
        self.label_page = QLabel(self.frame_8)
        self.label_page.setObjectName("last_btn")
        # self.label_page.clicked.connect(self.last_page)
        self.horizontalLayout_5.addWidget(self.label_page)
        self.label_page.setMaximumSize(100,45)
        self.label_page.setAlignment(Qt.AlignCenter)
        self.label_page.adjustSize()        
        
        self.pushButton_nex = QPushButton(self.frame_8)
        self.pushButton_nex.setObjectName("next_btn")
        self.pushButton_nex.clicked.connect(self.next_page)
        self.horizontalLayout_5.addWidget(self.pushButton_nex)
        

        self.horizontalLayout_2.addWidget(self.frame_down_right)


        self.treeView = QTreeWidget(self.groupBox_img)
        self.treeView.setMaximumSize(QSize(200, 16777215))
        self.treeView.setObjectName("treeView")
        self.treeView.setColumnCount(1)
        self.treeView.setHeaderLabels(["Directories"])
        self.treeView.setIconSize(QSize(20, 20))
        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.treeView.itemClicked.connect(self.on_update_folder)
        self.horizontalLayout_4.addWidget(self.treeView)
        



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
        self.actionExit.triggered.connect(self.close)
        # self.actionExit.triggered.connect(qApp.quit)
        self.actionOpen_Working_Folder = QAction(self)
        self.actionOpen_Working_Folder.setObjectName("actionOpen_Working_Folder")
        self.actionOpen_Working_Folder.triggered.connect(self.open_work_folder)
        self.actionOpen_Save_Folder = QAction(self)
        self.actionOpen_Save_Folder.setObjectName("actionOpen_Save_Folder")
        self.actionOpen_Save_Folder.triggered.connect(self.open_save_folder)
        self.actionSegmentation = QAction(self)
        self.actionSegmentation.setObjectName("Segmentation")
        self.actionSegmentation.triggered.connect(self.on_segmentation)
        self.actionCheck = QAction(self)
        self.actionCheck.setObjectName("actionCheck")
        self.actionCheck.triggered.connect(self.on_check)
        # self.actionProduce_DataSets = QAction(self)
        # self.actionProduce_DataSets.setObjectName("actionProduce_DataSets")
        # self.actionProduce_DataSets.triggered.connect(self.on_make_dataset)
        self.actionHelp = QAction(self)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.triggered.connect(self.on_help)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Working_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Save_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSegmentation)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionCheck)
        self.menuFile.addSeparator()
        # self.menuFile.addAction(self.actionProduce_DataSets)
        # self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionHelp)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi()
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("Segman", "Segman"))
        self.pushButton_seg.setText(_translate("Segman", "Segman"))
        self.pushButton_tag.setText(_translate("Segman", "Tagor"))
        self.pushButton_mod.setText(_translate("Segman", "Modler"))
        self.pushButton_sor.setText(_translate("Segman", "Sorta"))
        
        self.pushButton_start.setText(_translate("Segman", "Start Segmentation"))        
        self.pushButton_che.setText(_translate("Segman", "Check"))
        
        self.groupBox.setTitle(_translate("Segman", "Working Folder"))
        self.pushButton_wor.setText(_translate("Segman", "Open Working Folder"))
        self.groupBox_2.setTitle(_translate("Segman", "Save Folder"))
        self.pushButton_sav.setText(_translate("Segman", "Open Save Folder"))
        
        self.groupBox_3.setTitle(_translate("Segman", "Settings"))
        self.groupBox_31.setTitle(_translate("Segman", "Image Types"))
        self.radioButton_Flowcam.setText(_translate("Segman", "FlowCAM"))
        self.radioButton_Microscope.setText(_translate("Segman", "Microscope"))
        self.groupBox_32.setTitle(_translate("Segman", "Minimum Size"))
        self.groupBox_33.setTitle(_translate("Segman", "Lens Magnification"))
        self.groupBox_34.setTitle(_translate("Segman", "Resolution(Microscope)"))
        self.groupBox_35.setTitle(_translate("Segman", "Connectivity(Microscope)"))
        self.groupBox_36.setTitle(_translate("Segman", "Work on"))
        
        self.pushButton_las.setText(_translate("Segman", "Last"))
        self.pushButton_nex.setText(_translate("Segman", "Next"))        
        self.label_page.setText(_translate("Segman", "Page Number"))
        

        self.menuFile.setTitle(_translate("Segman", "File"))
        self.menuHelp.setTitle(_translate("Segman", "Help"))
        self.actionExit.setText(_translate("Segman", "Exit"))
        self.actionOpen_Working_Folder.setText(_translate("Segman", "Open Working Folder"))
        self.actionOpen_Save_Folder.setText(_translate("Segman", "Open Save Folder"))
        self.actionSegmentation.setText(_translate("Segman", "Start Segmentation"))
        self.actionCheck.setText(_translate("Segman", "Check Results"))
        # self.actionProduce_DataSets.setText(_translate("Segman", "Produce DataSets"))
        self.actionHelp.setText(_translate("Segman", "Help"))
        
    def open_work_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Load the work folder with images', '/')
        if file_path == None:
            QMessageBox.information(self,'Error','No work path selected')
        else:
            self.work_path = file_path
            self.tree_path = self.work_path
            self.work_paths.append(file_path)
            self.comboBox_wor.addItem(file_path.split("/")[-1])
            self.comboBox_wor.setCurrentText(file_path.split("/")[-1])
            
            self.treeView.clear()

            root = QTreeWidgetItem(self.treeView)
            root.setText(0,self.work_path.split('/')[-1])
            # root.setIcon(0,QIcon(icon))
            for _, dirs, _ in os.walk(self.work_path):
                self.CreateTree(dirs, root, self.work_path)
            QApplication.processEvents()

            self.image_show()

    def renew_work_folder(self):
        idx = self.comboBox_wor.currentIndex()
        self.work_path = self.work_paths[idx]
        self.tree_path = self.work_path
        self.treeView.clear()
        root = QTreeWidgetItem(self.treeView)
        root.setText(0,self.work_path.split('/')[-1])
        for _, dirs, _ in os.walk(self.work_path):
            self.CreateTree(dirs, root, self.work_path)
        QApplication.processEvents()
        # self.image_show()

    def open_save_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load Save Folder", "./")
        if file_path == None:
            QMessageBox.information(self,'Error','No save path selected')
        else:
            self.save_path = file_path
            self.save_paths.append(file_path)
            self.comboBox_sav.addItem(file_path.split("/")[-1])
            self.comboBox_sav.setCurrentText(file_path.split("/")[-1])

    def renew_save_folder(self):

        idx = self.comboBox_sav.currentIndex()
        self.save_path = self.save_paths[idx]

            
    def CreateTree(self, dirs, root, path):
        for dirname in dirs:
            path_new = path + '\\' + dirname
            if os.path.isdir(path_new):
                child = QTreeWidgetItem(root)
                child.setText(0,dirname)
                for _ , dirs_new, _ in os.walk(path_new):
                    self.CreateTree(dirs_new, child, path_new)

    
    def image_show(self):
        if self.tree_path:
            file_path = self.tree_path
            print('file_path: {}'.format(file_path))
        #     # print(file_path)
            img_type = ['png','jpg','tif','jpeg','bmp','jfif']
            self.clear_layout()
            # QApplication.processEvents()
            self.img_list = list(x for x in os.listdir(file_path) if x.split(".")[-1] in img_type)
            self.image_ids = list(str(file_path + '/' +x) for x in self.img_list)
            # print(self.img_list)
            num = len(self.img_list)

            self.num_page = 1
            self.label_page.setText("{:d} of {:d}".format(self.num_page, (len(self.img_list)-1) // self.num_a_page+1))
            self.select_ids = [False] * num
            if num < self.num_a_page:
                num_img = num
            else:
                num_img = self.num_a_page
            if num !=0:
                for i in range(num_img):
                    image_id = self.image_ids[(self.num_page - 1)*self.num_a_page + i]
                    # print(self.image_id)
                    select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                    pixmap = QPixmap(image_id)
                    self.addImage(pixmap, image_id, i, select_id)
                    print('load picture：', image_id)
                    QApplication.processEvents()
            else:
                # QMessageBox.warning(self,'Error','Empty folder!')
                self.clear_layout()
                self.event(exit())
        else:
            QMessageBox.warning(self, 'Error', 'Wrong with work path!')
    
    def addImage(self, pixmap, image_id, i, select_id):

        self.col = i % 4
        self.row = i // 4

        clickable_image = QClickableImage(self.displayed_image_size, 
                                          self.displayed_image_size, pixmap, image_id, select_id)
        clickable_image.leftClicked.connect(lambda:self.on_left_clicked(image_id))
        clickable_image.rightClicked.connect(self.on_right_clicked)
        clickable_image.doubleClicked.connect(lambda:self.on_double_clicked(image_id,pixmap))
        self.gridLayout_images.addWidget(clickable_image, self.row, self.col)
        # print(self.row, self.col)

    # def setDisplayedImageSize(self,image_size):
    #     self.displayed_image_size =image_size

    def last_page(self):
        # print("show last page")
        if self.num_page > 1:
            self.num_page -= 1
            self.label_page.setText("{:d} of {:d}".format(self.num_page, (len(self.img_list)-1) // self.num_a_page+1))
            if len(self.img_list) !=0:
                self.clear_layout()
                for i in range(self.num_a_page):
                    image_id = self.image_ids[(self.num_page - 1)*self.num_a_page + i]
                    # print(self.image_id)
                    select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                    pixmap = QPixmap(image_id)
                    self.addImage(pixmap, image_id, i, select_id)
                    print('load image：', image_id)
                    QApplication.processEvents()
    
    def next_page(self):
        # print("show next page")
        if (self.num_page > 0) and (self.num_page - 1 < (len(self.img_list)-1) // self.num_a_page):
            self.num_page += 1
            self.clear_layout()
            self.label_page.setText("{:d} of {:d}".format(self.num_page, (len(self.img_list)-1) // self.num_a_page+1))
            for i in range(self.num_a_page):
                image_id = self.image_ids[(self.num_page - 1)*self.num_a_page + i]
                # print(self.image_id)
                select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                pixmap = QPixmap(image_id)
                self.addImage(pixmap, image_id, i, select_id)
                print('load image：', image_id)
                QApplication.processEvents()
    
    def on_left_clicked(self, image_id):

        select = self.select_ids[self.image_ids.index(image_id)]
        if select == False:
            self.select_ids[self.image_ids.index(image_id)] = True
        else:
            self.select_ids[self.image_ids.index(image_id)] = False
        print('image id = ', image_id, " Selected:", not select)        

    def on_right_clicked(self):
        print('right clicked - image id = ')
        
    def on_double_clicked(self, img_id, pix):
        if img_id:
            self.child_window = Ui_Viewer(img_id, pix)
            self.child_window.show()
        print('double clicked - image id = ', img_id)
        
    def clear_layout(self):
        for i in range(self.gridLayout_images.count()):
            self.gridLayout_images.itemAt(i).widget().deleteLater()
                    
    def on_update_folder(self):
        item = self.treeView.selectedItems()[0]
        self.class_selected = item.text(0)
        texts = []
        while item is not None:
            texts.append(item.text(0))
            item = item.parent()
        if len(texts) >=2:
            texts = texts[:-1]
            texts.reverse()
            path = "/".join(texts)
            self.tree_path = self.work_path + "/" + path
        else:
            self.tree_path = self.work_path
        self.image_show()
        print("Class selected: ", self.class_selected)

    def on_imagetype(self):
        radiobutton = self.sender()
        if radiobutton.isChecked():
            if radiobutton.text() == "Microscope":
                self.imagetype = 0
            else:
                self.imagetype = 1
        self.label_status.setText("image type = %d " %self.imagetype)


    def on_connectivity(self):
        radiobutton = self.sender()
        if radiobutton.isChecked():
            if radiobutton.text() == "High":
                self.connectivity = 0
            elif radiobutton.text() == "Middle":
                self.connectivity = 1
            else:
                self.connectivity = 2
        self.label_status.setText("Connectivity Setting for microscope image = %d " %self.connectivity)        


    def on_level(self):

        radiobutton = self.sender()
        # print(radiobutton.objectName())
        if radiobutton.isChecked():
            if radiobutton.objectName() == "level0":
                self.level = 0
                if self.work_path:
                    self.label_status.setText("Work on current directory:%s" %self.work_path)
                else:
                    self.label_status.setText("Work on current directory")
            else:
                self.level = 1
                if self.work_path:
                    self.label_status.setText("Work on Sub-Directories of %s" %self.work_path)
                else:
                    self.label_status.setText("Work on current directory")

    def on_start_segman(self):
        print("seg")

    def on_start_tagor(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_modler(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_sorta(self):
        print("uncheck all selection, and clear the tags")
        
    def on_check(self):
        print("check the result")
        if self.imagetype == 0:
            check_path = self.save_path +"/COI/" 
        else:
            check_path = self.save_path +"/Cut/" 
        file_path = QFileDialog.getExistingDirectory(self, "Load Out Folder", check_path)
        self.label_status.setText("Check the results in directory: %s"%file_path)
        if file_path == None:
            QMessageBox.information(self,'Error','No check path selected')
        else:
            self.treeView.clear()
            # self.tree_path = file_path
            self.work_path = file_path
            root = QTreeWidgetItem(self.treeView)
            root.setText(0,file_path.split('/')[-1])
            # root.setIcon(0,QIcon(icon))
            for _, dirs, _ in os.walk(file_path):
                self.CreateTree(dirs, root, file_path)
            self.on_update_folder()
            QApplication.processEvents()


    def on_segmentation(self):
        self.label_status.setText("Start Segmentation!")
        if self.save_path and self.work_path:
            if self.imagetype == 0:
                self.microscope_cut()
            else:
                self.flowcam_cut()
            self.label_status.setText("Segmentation Finished!")
            self.label_progress.setText("")
        else:
            QMessageBox.warning(self,'Error','No work path or save path found!')
            self.label_status.setText("Segmentation Error")
            
    def microscope_cut(self):

        path=self.work_path
        folder_out = self.save_path
        imgtypes=['.tif', ".png",".jpg", ".gif", ".bmp", ".jpeg","jfif","jif", "jpe"]
        th_size = self.size_min / self.pixelsize
        if self.level == 0:
            count = 0
            countall = 0
            for fl in os.listdir(path):
                countall += 1
            print("Total Images: %d"%countall)
            for name in os.listdir(path):
                count += 1
                percent = count / countall
                imgtype = "." + name.split(".")[-1]
                if imgtype in imgtypes:
                    img_path=os.path.join(path,name)
                    self.label_status.setText(img_path)
                    path_save_COI=folder_out+'/COI/'
                    path_save=folder_out+'/Cut/'
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    if not os.path.exists(path_save_COI):
                        os.makedirs(path_save_COI)
                    try:                     
                        image = cv2.imread(img_path)
                        if image is None:
                            tmp = imageio.mimread(img_path)
                            if tmp is not None:
                                imt = np.array(tmp)
                                imt = imt[0]
                                image = imt[:,:,0:3]
                        imgs,image_coi=COI_Save(image,th_size)
                        cv2.imwrite(path_save_COI+name.replace(imgtype,'')+'_COI_rectangled.png',image_coi)
                        i=0
                        for img in imgs:
                            i+=1                   
                            img=np.array(img)
                            img=img[0,:,:,:]
                            cv2.imwrite(path_save+name.replace(imgtype,'')+'_COI_%03d.png'%i,img)               
                    except Exception as e:
                        self.label_status.setText('Wrong with opening image, and skipped: %s'%e)
                        print('Wrong with opening image, and skipped: %s'%e)
                        continue
                    self.label_progress.setText("%d of total %d images segmented, progressing %.02f %%"%(count,countall, percent))
                    print("%d of total %d images segmented, progressing %.02f %%"%(count,countall, percent))
                    QApplication.processEvents()
        else:
            countall= 0
            count = 0
            for _, _, fls in os.walk(path):
                countall += len(fls)
                
            for imgdir in os.listdir(path):
                data_path=path+"/"+imgdir+'/'
                if os.path.isdir(data_path):
                    for name in os.listdir(data_path):
                        imgtype = "." + name.split(".")[-1]
                        count += 1
                        percent = count / countall
                        if imgtype in imgtypes:
                            img_path=os.path.join(data_path,name)
                            self.label_status.setText(img_path)
                            path_save_COI=folder_out+'/COI/'+imgdir+'/'
                            path_save=folder_out+'/Cut/'+imgdir+'/'
                            #               path_save='C:/Python/Microscopic_Image/test_cut/'
                            if not os.path.exists(path_save):
                                os.makedirs(path_save)
                                
                            if not os.path.exists(path_save_COI):
                                # if self.imagetype == 0:
                                os.makedirs(path_save_COI)
                            try:                     
                                image = cv2.imread(img_path)
                                if image is None:
                                    tmp = imageio.mimread(img_path)
                                    if tmp is not None:
                                        imt = np.array(tmp)
                                        imt = imt[0]
                                        image = imt[:,:,0:3]
                                imgs,image_coi=COI_Save(image,th_size)
        
        
                                cv2.imwrite(path_save_COI+name.replace(imgtype,'')+'_'+imgdir+'_COI_rectangled.png',image_coi)
                                i=0
                                for img in imgs:
                                    i+=1                   
                                    img=np.array(img)
                                    img=img[0,:,:,:]
                                    cv2.imwrite(path_save+name.replace(imgtype,'')+'_'+imgdir+'_COI_%03d.png'%i,img)               
                            except Exception as e:
                                self.label_status.setText('Wrong with opening image, and skipped: %s'%e)
                                print('Wrong with opening image, and skipped: %s'%e)
                                continue
                            self.label_progress.setText("%d of total %d images segmented, progressing %.02f %%"%(count,countall, percent*100))
                            QApplication.processEvents()


    def flowcam_cut(self):
        filetype='.tif'
        flowcam_path = self.work_path
        target_path = self.save_path
        if self.level == 0: 
            count = 0
            countall = 0
            for fl in os.listdir(flowcam_path):
                countall += 1
            for fl in os.listdir(flowcam_path):
                count += 1
                percent = count / countall
                if (filetype in fl) & ('cal_image' not in fl):
                    # start = time.time()
                    tif_name=fl.replace(filetype,'')  #生成不带‘.tif’后缀的文件名
                    img_path = flowcam_path + "/" + fl
                    self.label_status.setText(img_path)
                    try:
                        img=cv2.imread(img_path)
                        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (1, 1),0)
                        (_, thresh) = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
                        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        for i in range(len(cnts)):
                            cnt=np.squeeze(cnts[i])
                            Xs = cnt[:,0]
                            Ys = cnt[:,1]
                            x1 = min(Xs)
                            x2 = max(Xs)
                            y1 = min(Ys)
                            y2 = max(Ys)
                            hight = y2 - y1
                            width = x2 - x1
                            crop_img = img[y1:y1+hight, x1:x1+width]
                            target_dir = target_path + "/Cut/"
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir)
                            cv2.imwrite(target_dir+'/'+tif_name+'%003d.png'%i, crop_img)
                    except Exception as e:
                        self.label_status.setText('Wrong with opening tif image, and skipped: %s'%e)
                        print('Wrong with opening tif image, and skipped: %s'%e)
                        continue
                    self.label_progress.setText("%d of total %d flowcam tifs segmented, progressing %.02f %%"%(count,countall, percent*100))
                    QApplication.processEvents()
        else:
            count = 0
            countall = 0
            for _, _, fls in os.walk(flowcam_path):
                countall += len(fls)
            for dirs in os.listdir(flowcam_path):
                sub_path=flowcam_path+"/"+dirs
                if os.path.isdir(sub_path):
                    # print(fls)
                    target_dir=target_path +"/Cut/" +dirs

                    for fl in os.listdir(sub_path):
                        count += 1
                        percent = count / countall
                        img_path = sub_path +"/" +fl
                        self.label_status.setText(img_path)
                        if (filetype in fl) & ('cal_image' not in fl):
                            # start = time.time()
                            tif_name=fl.replace(filetype,'')  #生成不带‘.tif’后缀的文件名

                            try: 
                                img=cv2.imread(img_path)
                                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                                blurred = cv2.GaussianBlur(gray, (1, 1),0)
                                (_, thresh) = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
                                cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                for i in range(len(cnts)):
                                    cnt=np.squeeze(cnts[i])
                                    Xs = cnt[:,0]
                                    Ys = cnt[:,1]
                                    x1 = min(Xs)
                                    x2 = max(Xs)
                                    y1 = min(Ys)
                                    y2 = max(Ys)
                                    hight = y2 - y1
                                    width = x2 - x1
                                    crop_img = img[y1:y1+hight, x1:x1+width]
                                    if not os.path.exists(target_dir):
                                        os.makedirs(target_dir)
                                    cv2.imwrite(target_dir+'/'+tif_name+'%003d.png'%i, crop_img)
                            except Exception as e:
                                self.label_status.setText('Wrong with opening tif image, and skipped: %s'%e)
                                print('Wrong with opening tif image, and skipped: %s'%e)
                                continue
                            self.label_progress.setText("%d of total %d flowcam tifs segmented, progressing %.02f %%"%(count,countall, percent*100))
                            QApplication.processEvents()
                    
    def on_help(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setText("Segman \nversion 1.0.0 \n\nThank for your interest! \n\nEmail:brigchen@xmu.edu.cn \ncopyright@brigchen \nXiamen University")
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        help_box.setFont(font)
        help_box.show()
        # des_path="./"
        
#%% 每一张图片，由图片，文件名组成；可单击和双击，双击打开图片查看器，单击选择，可多选    
class QClickableImage(QWidget):
    leftClicked = pyqtSignal()
    rightClicked = pyqtSignal()
    doubleClicked = pyqtSignal()
    def __init__(self, width = 0, height = 0, pixmap = None, image_id = '',  select = False):
        QWidget.__init__(self)
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.singleClicked)        
        self.layout =QVBoxLayout(self)
        self.label_img = QLabel()
        self.label_img.setObjectName('label_img')
        
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_img.sizePolicy().hasHeightForWidth())
        
        self.label_img.setSizePolicy(sizePolicy)
       
        self.label_name = QLabel()
        self.label_name.setObjectName('label_name')
        self.label_name.setMinimumSize(QSize(0, 40))
        self.label_name.setMaximumSize(QSize(width, 50))
        self.width = width
        self.height = height
        self.pixmap = pixmap
        self.img_selected = False

        if self.width and self.height:
            self.resize(self.width,self.height)
        
        # self.layout.addWidget(self.label_click)
        if self.pixmap:
            pixmap = self.pixmap.scaled(QSize(self.width,self.height),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.label_img.setPixmap(pixmap)
            self.label_img.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label_img)
            if select == False:
                self.label_img.setStyleSheet("")
                self.img_selected = False
            else:
                color = "background-color:#66CCFF"
                self.label_img.setStyleSheet(color)
                self.img_selected = True
                
        if image_id:
            self.image_id = image_id
            self.label_name.setText(image_id.split("/")[-1])
            self.label_name.setAlignment(Qt.AlignCenter)
            self.label_name.adjustSize()
            self.layout.addWidget(self.label_name)
        self.setLayout(self.layout)
   
    # 判断左右点击与双击
    def singleClicked(self): 
        if self.timer.isActive():
            self.timer.stop()
            if self.img_selected == False:
                color = "background-color:#66CCFF"
                self.label_img.setStyleSheet(color)
                self.img_selected = True
                # self.select_ids[self.image_ids.index(image_id)] = True
            else:
                self.label_img.setStyleSheet("")
                self.img_selected = False
                # self.select_ids[self.image_ids.index(image_id)] = False
            # QApplication.processEvents()
            self.leftClicked.emit() 
            print(self.img_selected)
    
    def mousePressEvent(self, event):           
        if event.buttons () == Qt.LeftButton:                           # 左键按下
            self.timer.start(300)
        elif event.buttons () == Qt.RightButton:                        # 右键按下
            self.rightClicked.emit()

    def mouseDoubleClickEvent(self, event):
        self.timer.stop()
        self.doubleClicked.emit()
#%%   
if __name__ =='__main__':
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    app = QApplication(sys.argv) 
    gui = Ui_Segman()
    gui.setWindowIcon(QIcon('./Segman_icon_small.png'))
    gui.showMaximized()
    sys.exit(app.exec_())  
