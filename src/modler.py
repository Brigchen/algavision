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
# from imagecut import COI_Save, Flowcam_Cut
# import pyqtgraph as pg

#%%
class Ui_modler(QMainWindow):
    def __init__(self):
        super(Ui_modler, self).__init__()
        self.dataset_paths = []
        self.dataset_path = ""
        self.output_paths = []
        self.output_path =""
        self.modeltypes = ["EfficientNet B0", "EfficientNet B1","EfficientNet B2","EfficientNet B3",
                          "EfficientNet B4","EfficientNet B5","EfficientNet B6","EfficientNet B7",
                          "EfficientNet B8","DenseNet 161","DenseNet 201","Customed"]
        self.mtype = self.modeltypes[3]
        self.ptype = 1
        
        self.loaded_full_model = ""
        self.loaded_full_models = []
        self.loaded_state_dict = ""
        self.loaded_state_dicts = []
        
        self.tree_path = ""
        self.select_ids = []
        self.image_ids = []
        self.image_list = []
        self.displayed_image_size = 300
        self.col = 0
        self.row = 0
        self.num_a_page = 12
#%% Hyper parameters settings
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.decay_weight = 0.01
        self.dropout = 0.5
        self.optimizer = ["SGD", "Adam", ""]
        self.criterion = ["CrossEntropyLoss"]
        self.earlystop = 10
        self.epochs = 30
        
#%% Setup UI
        self.setupUi()

    def setupUi(self):
        self.setObjectName("modler")
        self.setWindowTitle('Modler: Deep learning training tools for plankton recognization')
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
        self.label.setPixmap(QPixmap("Modler_logo2.png"))
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
        self.gridLayout_top.setContentsMargins(0, 0, 10, 0)
        self.gridLayout_top.setHorizontalSpacing(15)
        self.gridLayout_top.setVerticalSpacing(0)
        self.gridLayout_top.setObjectName("gridLayout_top")
        
        self.pushButton_seg = QPushButton(self.frame_top_right)
        self.pushButton_seg.setObjectName("modler")
        self.pushButton_seg.clicked.connect(self.on_start_modler)
        self.gridLayout_top.addWidget(self.pushButton_seg, 0, 0, 1, 1)
        
        self.pushButton_tag = QPushButton(self.frame_top_right)
        self.pushButton_tag.setObjectName("modler")
        self.pushButton_tag.clicked.connect(self.on_start_tagor)
        self.gridLayout_top.addWidget(self.pushButton_tag, 0, 1, 1, 1)
        
        self.pushButton_mod = QPushButton(self.frame_top_right)
        self.pushButton_mod.setObjectName("modler")
        self.pushButton_mod.clicked.connect(self.on_start_modler)
        self.gridLayout_top.addWidget(self.pushButton_mod, 0, 2, 1, 1)
        
        self.pushButton_sor = QPushButton(self.frame_top_right)
        self.pushButton_sor.setObjectName("sorta")
        self.pushButton_sor.clicked.connect(self.on_start_modler)
        self.gridLayout_top.addWidget(self.pushButton_sor, 0, 3, 1, 1)
        
        self.pushButton_enhance = QPushButton(self.frame_top_right)
        self.pushButton_enhance.setObjectName("dataset")
        self.pushButton_enhance.clicked.connect(self.on_enhancement)
        self.gridLayout_top.addWidget(self.pushButton_enhance, 1, 0, 1, 2)    
                       
        self.pushButton_start = QPushButton(self.frame_top_right)
        self.pushButton_start.setObjectName("start")
        self.pushButton_start.clicked.connect(self.on_training)
        self.gridLayout_top.addWidget(self.pushButton_start, 1, 2, 1, 2)
        
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
        self.comboBox_mtype.setCurrentIndex(5)
        self.comboBox_mtype.currentIndexChanged.connect(self.on_mtype)
        self.verticalLayout_mtype = QVBoxLayout(self.groupBox_31)
        self.verticalLayout_mtype.addWidget(self.comboBox_mtype)

        self.groupBox_32 = QGroupBox(self.groupBox_mode)    #min size
        self.groupBox_32.setMinimumSize(QSize(200, 75))
        self.groupBox_32.setMaximumSize(QSize(500, 150))
        self.verticalLayout_mode.addWidget(self.groupBox_32)
        self.radioButton_full_struct = QRadioButton(self.groupBox_32)
        self.radioButton_full_struct.setObjectName("full_struct")
        self.radioButton_full_struct.setText("Full Structure")
        self.radioButton_state_dict = QRadioButton(self.groupBox_32)
        self.radioButton_state_dict.setObjectName("state_dict")
        self.radioButton_state_dict.setText("State Dict")
        self.radioButton_state_dict.setChecked(True)
        self.verticalLayout_ptype = QVBoxLayout(self.groupBox_32)
        self.verticalLayout_ptype.addWidget(self.radioButton_full_struct)
        self.verticalLayout_ptype.addWidget(self.radioButton_state_dict)
        self.radioButton_full_struct.toggled.connect(self.on_ptype)
        self.radioButton_state_dict.toggled.connect(self.on_ptype)

        self.groupBox_33 = QGroupBox(self.groupBox_mode)    #magnification
        self.groupBox_33.setObjectName("Objective lens magnification")
        self.groupBox_33.setMinimumSize(QSize(200, 75))
        self.groupBox_33.setMaximumSize(QSize(500, 100))
        self.verticalLayout_mode.addWidget(self.groupBox_33)
        self.verticalLayout_fload = QVBoxLayout(self.groupBox_33)
        self.pushButton_load_full = QPushButton(self.groupBox_33)
        self.pushButton_load_full.setObjectName("Plankton")
        self.verticalLayout_fload.addWidget(self.pushButton_load_full )
        self.pushButton_load_full .clicked.connect(self.load_full_model)
        self.pushButton_load_full.setText("Load Full Model")
        if self.ptype == 1:
            self.pushButton_load_full.setEnabled(False) 
        
        self.comboBox_load_full = QComboBox(self.groupBox_33)
        self.comboBox_load_full.setObjectName("load_full")
        self.comboBox_load_full.currentIndexChanged.connect(self.renew_full_model) 
        self.verticalLayout_fload.addWidget(self.comboBox_load_full)
        
        self.groupBox_34 = QGroupBox(self.groupBox_mode)    #resolution
        self.groupBox_34.setObjectName("Pixel size")
        self.groupBox_34.setMinimumSize(QSize(200, 75))
        self.groupBox_34.setMaximumSize(QSize(500, 100))
        self.verticalLayout_mode.addWidget(self.groupBox_34)
        self.verticalLayout_sload = QVBoxLayout(self.groupBox_34)
                
        self.pushButton_load_state = QPushButton(self.groupBox_34)
        self.pushButton_load_state.setObjectName("load state")
        self.verticalLayout_sload.addWidget(self.pushButton_load_state)
        self.pushButton_load_state.clicked.connect(self.load_state_model)
        self.pushButton_load_state.setText("Load State Dict")
        if self.ptype == 0:
            self.pushButton_load_state.setEnabled(False) 
        
        self.comboBox_load_state = QComboBox(self.groupBox_34)
        self.comboBox_load_state.setObjectName("load_state")
        self.comboBox_load_state.currentIndexChanged.connect(self.renew_state_model) 
        self.verticalLayout_sload.addWidget(self.comboBox_load_state)

        self.groupBox_36 = QGroupBox(self.groupBox_mode)    #magnification
        self.groupBox_36.setObjectName("levels")
        self.verticalLayout_mode.addWidget(self.groupBox_36)
        self.groupBox_36.setSizePolicy(sizePolicy)

        self.frame_down_mid = QFrame(self.frame_down)
        self.frame_down_mid.setMinimumSize(QSize(250, 0))
        self.frame_down_mid.setMaximumSize(QSize(350, 16777215))
        self.frame_down_mid.setFrameShape(QFrame.StyledPanel)
        self.frame_down_mid.setFrameShadow(QFrame.Raised)
        self.frame_down_mid.setObjectName("frame_down_mid")
        self.horizontalLayout_down.addWidget(self.frame_down_mid)
        self.verticalLayout_hyper = QVBoxLayout(self.frame_down_mid)
        self.verticalLayout_hyper.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_hyper.setSpacing(10)

        self.groupBox_hyper = QGroupBox(self.frame_down_mid)                        #
        self.groupBox_hyper.setTitle("Hyper-Paramenters")
        self.groupBox_hyper.setSizePolicy(sizePolicy)
        self.verticalLayout_hyper.addWidget(self.groupBox_hyper)
        self.verticalLayout_h1 = QVBoxLayout(self.groupBox_hyper)

        doubleValidator=QDoubleValidator()
        doubleValidator.setRange(0,1)
        doubleValidator.setNotation(QDoubleValidator.StandardNotation)
        doubleValidator.setDecimals(5)

        intValidator=QIntValidator()
        intValidator.setRange(1,2000)

        self.groupBox_41 = QGroupBox(self.groupBox_hyper)
        self.groupBox_41.setTitle("Learning Rate")
        self.verticalLayout_h1.addWidget(self.groupBox_41)
        self.lineEdit_lr = QLineEdit(self.groupBox_41)
        self.lineEdit_lr.setValidator(doubleValidator)
        self.lineEdit_lr.setAlignment(Qt.AlignCenter)
        self.lineEdit_lr.setText(str(self.learning_rate))
        self.verticalLayout_41 = QVBoxLayout(self.groupBox_41)
        self.verticalLayout_41.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_41.addWidget(self.lineEdit_lr)

        self.groupBox_42 = QGroupBox(self.groupBox_hyper)
        self.groupBox_42.setTitle("Momentum")
        self.verticalLayout_h1.addWidget(self.groupBox_42)
        self.lineEdit_mm = QLineEdit(self.groupBox_42)
        self.lineEdit_mm.setValidator(doubleValidator)
        self.lineEdit_mm.setAlignment(Qt.AlignCenter)
        self.lineEdit_mm.setText(str(self.momentum))
        self.verticalLayout_42 = QVBoxLayout(self.groupBox_42)
        self.verticalLayout_42.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_42.addWidget(self.lineEdit_mm)

        self.groupBox_43 = QGroupBox(self.groupBox_hyper)
        self.groupBox_43.setTitle("Decay Weight")
        self.verticalLayout_h1.addWidget(self.groupBox_43)
        self.lineEdit_dw = QLineEdit(self.groupBox_43)
        self.lineEdit_dw.setValidator(doubleValidator)
        self.lineEdit_dw.setAlignment(Qt.AlignCenter)
        self.lineEdit_dw.setText(str(self.decay_weight))
        self.verticalLayout_43 = QVBoxLayout(self.groupBox_43)
        self.verticalLayout_43.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_43.addWidget(self.lineEdit_dw)

        self.groupBox_44 = QGroupBox(self.groupBox_hyper)
        self.groupBox_44.setTitle("Dropout")
        self.verticalLayout_h1.addWidget(self.groupBox_44)
        self.lineEdit_do = QLineEdit(self.groupBox_44)
        self.lineEdit_do.setValidator(doubleValidator)
        self.lineEdit_do.setAlignment(Qt.AlignCenter)
        self.lineEdit_do.setText(str(self.dropout))
        self.verticalLayout_44 = QVBoxLayout(self.groupBox_44)
        self.verticalLayout_44.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_44.addWidget(self.lineEdit_do)

        self.groupBox_45 = QGroupBox(self.groupBox_hyper)
        self.groupBox_45.setTitle("Criterion")
        self.verticalLayout_h1.addWidget(self.groupBox_45)
        self.comboBox_ct = QComboBox(self.groupBox_45)
        self.comboBox_ct.addItems(self.criterion)
        self.comboBox_ct.setCurrentIndex(0)
        self.verticalLayout_45 = QVBoxLayout(self.groupBox_45)
        self.verticalLayout_45.addWidget(self.comboBox_ct)
        
        self.groupBox_46 = QGroupBox(self.groupBox_hyper)
        self.groupBox_46.setTitle("Optimizer")
        self.verticalLayout_h1.addWidget(self.groupBox_46)
        self.comboBox_op = QComboBox(self.groupBox_46)
        self.comboBox_op.addItems(self.optimizer)
        self.comboBox_op.setCurrentIndex(0)
        self.verticalLayout_46 = QVBoxLayout(self.groupBox_46)
        self.verticalLayout_46.addWidget(self.comboBox_op)
        
        self.groupBox_47 = QGroupBox(self.groupBox_hyper)
        self.groupBox_47.setTitle("epochs")
        self.verticalLayout_h1.addWidget(self.groupBox_47)
        self.lineEdit_ep = QLineEdit(self.groupBox_44)
        self.lineEdit_ep.setValidator(intValidator)
        self.lineEdit_ep.setAlignment(Qt.AlignCenter)
        self.lineEdit_ep.setText(str(self.epochs))
        self.verticalLayout_47 = QVBoxLayout(self.groupBox_47)
        self.verticalLayout_47.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_47.addWidget(self.lineEdit_ep)

        self.groupBox_48 = QGroupBox(self.groupBox_hyper)
        self.groupBox_48.setTitle("Optimizer")
        self.verticalLayout_h1.addWidget(self.groupBox_48)
        self.lineEdit_es = QLineEdit(self.groupBox_44)
        self.lineEdit_es.setValidator(intValidator)
        self.lineEdit_es.setAlignment(Qt.AlignCenter)
        self.lineEdit_es.setText(str(self.earlystop))
        self.verticalLayout_48 = QVBoxLayout(self.groupBox_48)
        self.verticalLayout_48.setContentsMargins(50, 15, 50, 15)
        self.verticalLayout_48.addWidget(self.lineEdit_es)

        self.groupBox_hyper2 = QGroupBox(self.frame_down_mid)    #
        self.verticalLayout_hyper.addWidget(self.groupBox_hyper2)
        self.verticalLayout_h2 = QVBoxLayout(self.groupBox_hyper2)
        self.groupBox_hyper2.setSizePolicy(sizePolicy)

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
        self.groupBox_status.setMinimumHeight(100)
        self.groupBox_status.setMaximumSize(5000,200)
        self.groupBox_status.setTitle("Status")

        self.label_status = QLabel(self.groupBox_status)
        self.label_status.setText("Welcome to use Modler tool for plankton deeping learning model training")
        self.verticalLayout_status = QVBoxLayout(self.groupBox_status)
        self.verticalLayout_status.addWidget(self.label_status)

        self.label_progress = QLabel(self.groupBox_status)
        self.verticalLayout_status.addWidget(self.label_progress)

        self.groupBox_plot = QGroupBox(self.frame_down_mid2)
        self.groupBox_plot.setTitle("Visualization")
        self.verticalLayout_view.addWidget(self.groupBox_plot)

        self.frame_plot1 = QFrame(self.groupBox_plot)
        self.verticalLayout_plot = QVBoxLayout(self.groupBox_plot)
        self.verticalLayout_plot.setContentsMargins(0, 0, 0, 0)

        self.plot1 = QGraphicsView(self.groupBox_plot)
        self.plot1.setStyleSheet("background:transparent;border:0px;")
        scene = QGraphicsScene()
        imgShow = QPixmap()
        imgShow.load("Modler_logo2.png")
        imgShowItem = QGraphicsPixmapItem()
        # imgShowItem.setPixmap(QPixmap(imgShow))
        imgShowItem.setPixmap(QPixmap(imgShow).scaled(800, 300))
        scene.addItem(imgShowItem)
        self.plot1.setScene(scene)
        # self.plot1.fitInView(QGraphicsPixmapItem(QPixmap(imgShow)))   
        # self.plot1.setScene(scene)
        # self.plotWdgt = pg.PlotWidget()
        # self.plotWdgt.setBackground('w')
        # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # plot_item = self.plotWdgt.plot(data)

        self.verticalLayout_plot.addWidget(self.plot1)

        self.plot2 = QGraphicsView(self.groupBox_plot)
        self.plot2.setStyleSheet("background:transparent;border:0px;")
        # self.plot2.setBackground('w')
        self.verticalLayout_plot.addWidget(self.plot2)

#%% Down right(last) frame for class name list

        self.frame_down_right = QFrame(self.frame_down)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_down_right.sizePolicy().hasHeightForWidth())
        self.frame_down_right.setSizePolicy(sizePolicy)
        self.frame_down_right.setFrameShape(QFrame.StyledPanel)
        self.frame_down_right.setFrameShadow(QFrame.Raised)
        self.frame_down_right.setMaximumSize(QSize(250, 16777215))

        self.horizontalLayout_down.addWidget(self.frame_down_right)
        self.verticalLayout_right = QVBoxLayout(self.frame_down_right)
        self.verticalLayout_right.setContentsMargins(0, 0, 0, 0)

        self.groupBox_classname = QGroupBox(self.frame_down_right)
        self.groupBox_classname.setTitle("List of Classes")
        self.verticalLayout_right.addWidget(self.groupBox_classname)
        self.verticalLayout_classname = QVBoxLayout(self.groupBox_classname)
        
        self.pushButton_load_class = QPushButton(self.groupBox_classname)
        self.pushButton_load_class.setText("Load list of class names")
        self.verticalLayout_classname.addWidget(self.pushButton_load_class)
        
        self.comboBox_load_class = QComboBox(self.groupBox_classname)
        self.verticalLayout_classname.addWidget(self.comboBox_load_class)
        
        self.treeView = QTreeWidget(self.groupBox_classname)
        self.treeView.setMaximumSize(QSize(250, 16777215))
        self.treeView.setObjectName("class list")
        self.treeView.setColumnCount(1)
        self.treeView.setHeaderLabels(["Classes"])
        self.treeView.setIconSize(QSize(20, 20))
        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        # self.treeView.itemClicked.connect(self.on_update_folder)
        self.verticalLayout_classname.addWidget(self.treeView)
        
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
        self.actionOpen_Dataset_Folder = QAction(self)
        self.actionOpen_Dataset_Folder.setObjectName("actionOpen_Dataset_Folder")
        self.actionOpen_Dataset_Folder.triggered.connect(self.open_dataset_folder)
        self.actionOpen_Output_Folder = QAction(self)
        self.actionOpen_Output_Folder.setObjectName("actionOpen_Output_Folder")
        self.actionOpen_Output_Folder.triggered.connect(self.open_output_folder)
        self.actionEnhancement = QAction(self)
        self.actionEnhancement.setObjectName("Make_train_test_datasets")
        self.actionEnhancement.triggered.connect(self.on_enhancement)
        self.actionTraining = QAction(self)
        self.actionTraining.setObjectName("actionTraining")
        self.actionTraining.triggered.connect(self.on_training)
        # self.actionProduce_DataSets = QAction(self)
        # self.actionProduce_DataSets.setObjectName("actionProduce_DataSets")
        # self.actionProduce_DataSets.triggered.connect(self.on_make_dataset)
        self.actionHelp = QAction(self)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.triggered.connect(self.on_help)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Dataset_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Output_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionEnhancement)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionTraining)
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
        self.setWindowTitle(_translate("modler", "Modler"))
        self.pushButton_seg.setText(_translate("modler", "Segman"))
        self.pushButton_tag.setText(_translate("modler", "Tagor"))
        self.pushButton_mod.setText(_translate("modler", "Modler"))
        self.pushButton_sor.setText(_translate("modler", "Sorta"))
        
        self.pushButton_enhance.setText(_translate("modler", "Dataset Enhancement"))        
        self.pushButton_start.setText(_translate("modler", "Start Training"))
        
        self.groupBox_dataset.setTitle(_translate("modler", "Plankton Dataset"))
        self.pushButton_dataset.setText(_translate("modler", "Open Classified Dataset Folder"))
        self.groupBox_output.setTitle(_translate("modler", "Output Folder"))
        self.pushButton_output.setText(_translate("modler", "Open Output Folder"))
        
        self.groupBox_mode.setTitle(_translate("modler", "Model Settings"))
        self.groupBox_31.setTitle(_translate("modler", "Model Types"))
        self.groupBox_32.setTitle(_translate("modler", "Model Parameter Type"))
        self.groupBox_33.setTitle(_translate("modler", "Pretrained Full Model"))
        self.groupBox_34.setTitle(_translate("modler", "Pretrained State Dict"))
        # self.groupBox_35.setTitle(_translate("modler", "Connectivity(Microscope)"))
        # self.groupBox_36.setTitle(_translate("modler", "empty"))
       

        self.menuFile.setTitle(_translate("modler", "File"))
        self.menuHelp.setTitle(_translate("modler", "Help"))
        self.actionExit.setText(_translate("modler", "Exit"))
        self.actionOpen_Dataset_Folder.setText(_translate("modler", "Open Datasets Folder"))
        self.actionOpen_Output_Folder.setText(_translate("modler", "Open Output Folder"))
        self.actionEnhancement.setText(_translate("modler", "Dataset Enhancement"))
        self.actionTraining.setText(_translate("modler", "Training"))
        # self.actionProduce_DataSets.setText(_translate("modler", "Produce DataSets"))
        self.actionHelp.setText(_translate("modler", "Help"))
        
    def open_dataset_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Load the folder with classified datasets images', '/')
        if file_path == None:
            print('Error','No work path selected')
        else:
            self.dataset_path = file_path
            self.tree_path = self.dataset_path
            self.dataset_paths.append(file_path)
            self.comboBox_dataset.addItem(file_path.split("/")[-1])
            self.comboBox_dataset.setCurrentText(file_path.split("/")[-1])
            
            self.treeView.clear()

            root = QTreeWidgetItem(self.treeView)
            root.setText(0,self.dataset_path.split('/')[-1])
            # root.setIcon(0,QIcon(icon))
            for _, dirs, _ in os.walk(self.dataset_path):
                self.CreateTree(dirs, root, self.dataset_path)
            QApplication.processEvents()

            self.image_show()

    def renew_dataset_folder(self):
        idx = self.comboBox_dataset.currentIndex()
        self.dataset_path = self.dataset_paths[idx]
        self.tree_path = self.dataset_path
        # self.treeView.clear()
        # root = QTreeWidgetItem(self.treeView)
        # root.setText(0,self.dataset_path.split('/')[-1])
        # for _, dirs, _ in os.walk(self.dataset_path):
        #     self.CreateTree(dirs, root, self.dataset_path)
        QApplication.processEvents()
        # self.image_show()

    def open_output_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load Output Folder", "./")
        if file_path == None:
            print('Error','No save path selected')
        else:
            self.output_path = file_path
            self.output_paths.append(file_path)
            self.comboBox_output.addItem(file_path.split("/")[-1])
            self.comboBox_output.setCurrentText(file_path.split("/")[-1])

    def renew_output_folder(self):

        idx = self.comboBox_output.currentIndex()
        self.output_path = self.output_paths[idx]


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
            self.tree_path = self.dataset_path + "/" + path
        else:
            self.tree_path = self.dataset_path
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


    def load_full_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'choose a model file with full structure', r'./Models',
                                               'model files(*.pt *.pth *.pkl)')
        name = fname.split("/")
        self.loaded_full_model = str(fname)
        self.loaded_full_models.append(str(fname))
        self.comboBox_load_full.addItem(name[-1])
        self.comboBox_load_full.setCurrentText(name[-1])
        self.label_status.setText("Loaded full model:%s"%name[-1])
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # model_name='efficientnet-b5' ####change
        # self.model = EfficientNet.from_pretrained(self.model_name)#, num_classes = self.taxa_num)
        # self.model._fc.out_features = self.taxa_num
        # if not use_gpu:
        #     self.model.load_state_dict(torch.load(self.Loaded_Model_Path, map_location='cpu'))
        # else:
        #     self.model.load_state_dict(torch.load(self.Loaded_Model_Path))
        # self.model = self.model.to(device)
        # self.result_text.append("加载模型已完成")

    def renew_full_model(self):
        idx = self.comboBox_load_full.currentIndex()
        self.loaded_full_model = self.loaded_full_models[idx]
        self.label_status.setText("Loaded full model:%s"%self.comboBox_load_full.currentText())

    def load_state_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'choose a model file with state dict', r'./Models',
                                               'model files(*.pt *.pth *.pkl)')
        name = fname.split("/")
        self.loaded_state_dict = str(fname)
        self.loaded_state_dicts.append(str(fname))
        self.comboBox_load_state.addItem(name[-1])
        self.comboBox_load_state.setCurrentText(name[-1])
        self.label_status.setText("Loaded state dict of model: %s"%name[-1])

    def renew_state_model(self):
        idx = self.comboBox_load_state.currentIndex()
        self.loaded_state_dict = self.loaded_state_dicts[idx]
        self.label_status.setText("Loaded state dict of model:%s"%self.comboBox_load_state.currentText())

    def on_mtype(self):
        combobox = self.sender()
        self.mtype = self.modeltypes[combobox.currentIndex()]
        self.label_status.setText("Model type: %s selected" %self.mtype)

    def on_ptype(self):
        radiobutton = self.sender()
        if radiobutton.isChecked():
            self.label_status.setText(radiobutton.objectName())
            if radiobutton.objectName() == "full_struct":
                self.ptype = 0
                self.pushButton_load_full.setEnabled(True)
                self.pushButton_load_state.setEnabled(False)
            else:
                self.ptype = 1
                self.pushButton_load_state.setEnabled(True)
                self.pushButton_load_full.setEnabled(False)

    def on_start_modler(self):
        print("seg")

    def on_start_tagor(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_modler(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_sorta(self):
        print("uncheck all selection, and clear the tags")
        
    def on_training(self):
        print("check the result")


    def on_enhancement(self):
        # print("used the class folders to make a train dataset and test dataset")
                # if img_id:
        # if self.output_path and self.dataset_path:
        self.child_window_2 = Ui_Enhance(self.output_path)
        self.child_window_2.setWindowIcon(QIcon('./Modler_icon_small.png'))
        self.child_window_2.show()
        # else:
        #     QMessageBox.warning(self,'Error','No class path found!')
            
    def microscope_cut(self):

        path=self.dataset_path
        folder_out = self.output_path
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
        flowcam_path = self.dataset_path
        target_path = self.output_path
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
        help_box.setText("Modler \nversion 1.0.0 \n\nThank for your interest! \n\nEmail:brigchen@xmu.edu.cn \ncopyright@brigchen \nXiamen University")
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
    gui = Ui_modler()
    gui.setWindowIcon(QIcon('./modler_icon_small.png'))
    gui.showMaximized()
    sys.exit(app.exec_())  
