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

#%%
class Ui_Tagor(QMainWindow):
    def __init__(self):
        super(Ui_Tagor, self).__init__()
        self.work_paths = []
        self.work_path = ""
        self.class_paths = []
        self.class_path =""
        self.tag_list = []
        self.tag_ids = []
        self.tag_selected = ""
        self.select_ids = []
        self.image_ids = []
        self.image_list = []
        self.displayed_image_size = 300
        self.col = 0
        self.row = 0
        self.num_a_page = 12
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Tagor")
        self.setWindowTitle('Tagor: tagging images')
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
        self.frame = QFrame(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QSize(0, 100))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QLabel(self.frame)
        self.label.setMaximumSize(QSize(300, 100))
        # self.label.setText("brigchen@xmu.edu.cn")
        self.label.setPixmap(QPixmap("Taqor_logo3.png"))
        self.label.setScaledContents(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.frame_4 = QFrame(self.frame)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_2 = QGridLayout(self.frame_4)
        self.gridLayout_2.setContentsMargins(0, 0, 10, 0)
        self.gridLayout_2.setHorizontalSpacing(15)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        self.pushButton_seg = QPushButton(self.frame_4)
        self.pushButton_seg.setObjectName("segman")
        self.pushButton_seg.clicked.connect(self.on_start_segman)
        self.gridLayout_2.addWidget(self.pushButton_seg, 0, 0, 1, 1)
        
        self.pushButton_tag = QPushButton(self.frame_4)
        self.pushButton_tag.setObjectName("tagor")
        # self.pushButton_seg.clicked.connect(self.on_start_tagor)
        self.gridLayout_2.addWidget(self.pushButton_tag, 0, 1, 1, 1)
        
        self.pushButton_mod = QPushButton(self.frame_4)
        self.pushButton_mod.setObjectName("modler")
        self.pushButton_mod.clicked.connect(self.on_start_modler)
        self.gridLayout_2.addWidget(self.pushButton_mod, 0, 2, 1, 1)
        
        self.pushButton_sor = QPushButton(self.frame_4)
        self.pushButton_sor.setObjectName("sorta")
        self.pushButton_sor.clicked.connect(self.on_start_modler)
        self.gridLayout_2.addWidget(self.pushButton_sor, 0, 3, 1, 1)
        
        self.pushButton_cpy = QPushButton(self.frame_4)
        self.pushButton_cpy.setObjectName("copy")
        self.pushButton_cpy.clicked.connect(self.on_copy_files)
        self.gridLayout_2.addWidget(self.pushButton_cpy, 1, 0, 1, 2)    
                       
        self.pushButton_dat = QPushButton(self.frame_4)
        self.pushButton_dat.setObjectName("datasets")
        self.pushButton_dat.clicked.connect(self.on_make_dataset)
        self.gridLayout_2.addWidget(self.pushButton_dat, 1, 2, 1, 2)
        
        self.horizontalLayout.addWidget(self.frame_4)
       
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.frame_2 = QFrame(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 10)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.frame_5 = QFrame(self.frame_2)
        self.frame_5.setMinimumSize(QSize(200, 0))
        self.frame_5.setMaximumSize(QSize(250, 16777215))
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        
        self.verticalLayout = QVBoxLayout(self.frame_5)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.groupBox = QGroupBox(self.frame_5)
        self.groupBox.setMinimumSize(QSize(0, 90))
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
        self.groupBox_2 = QGroupBox(self.frame_5)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("class")
        
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_cla = QPushButton(self.groupBox_2)
        self.pushButton_cla.setObjectName("class_btn")
        self.verticalLayout_3.addWidget(self.pushButton_cla)
        self.pushButton_cla.clicked.connect(self.open_class_folder)
        
        self.comboBox_cla = QComboBox(self.groupBox_2)
        self.comboBox_cla.setObjectName("class_cb")
        self.comboBox_cla.currentIndexChanged.connect(self.renew_class_folder) 
        self.verticalLayout_3.addWidget(self.comboBox_cla)
        
        self.treeView = QTreeWidget(self.groupBox_2)
        self.treeView.setMaximumSize(QSize(250, 16777215))
        self.treeView.setObjectName("treeView")
        self.treeView.setColumnCount(1)
        self.treeView.setHeaderLabels(["Classes"])
        self.treeView.setIconSize(QSize(20, 20))
        self.treeView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.treeView.itemClicked.connect(self.on_check_class)
        
        # self.treeView.setColumnWidth(0, 250)
        self.verticalLayout_3.addWidget(self.treeView)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.horizontalLayout_2.addWidget(self.frame_5)
        self.frame_6 = QFrame(self.frame_2)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        
        self.verticalLayout_5 = QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        
        self.scrollArea = QScrollArea(self.frame_6)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("image_show_area")
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 772, 560))
        self.scrollAreaWidgetContents.setObjectName("image_show")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_images = QGridLayout(self.scrollAreaWidgetContents)

        self.verticalLayout_5.addWidget(self.scrollArea)
        self.frame_8 = QFrame(self.frame_6)
        self.frame_8.setMinimumSize(QSize(0, 45))
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        
        self.horizontalLayout_3 = QHBoxLayout(self.frame_8)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.pushButton_las = QPushButton(self.frame_8)
        self.pushButton_las.setObjectName("last_btn")
        self.pushButton_las.clicked.connect(self.last_page)
        self.horizontalLayout_3.addWidget(self.pushButton_las)
        
        self.label_page = QLabel(self.frame_8)
        self.label_page.setObjectName("last_btn")
        # self.label_page.clicked.connect(self.last_page)
        self.horizontalLayout_3.addWidget(self.label_page)
        self.label_page.setMaximumSize(100,45)
        self.label_page.setAlignment(Qt.AlignCenter)
        self.label_page.adjustSize()        
        
        self.pushButton_nex = QPushButton(self.frame_8)
        self.pushButton_nex.setObjectName("next_btn")
        self.pushButton_nex.clicked.connect(self.next_page)
        self.horizontalLayout_3.addWidget(self.pushButton_nex)
        
        self.verticalLayout_5.addWidget(self.frame_8)
        self.horizontalLayout_2.addWidget(self.frame_6)
        self.groupBox_3 = QGroupBox(self.frame_2)
        self.groupBox_3.setMaximumSize(QSize(250, 3000))
        self.groupBox_3.setObjectName("tag_list")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_3)
        
        self.verticalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_4.setSpacing(5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        
        self.pushButton_lis = QPushButton(self.groupBox_3)
        self.pushButton_lis.setObjectName("tag_btn")
        self.verticalLayout_4.addWidget(self.pushButton_lis)
        self.pushButton_lis.clicked.connect(self.load_tag_list)
        
        self.pushButton_add = QPushButton(self.groupBox_3)
        self.pushButton_add.setObjectName("add_btn")
        self.verticalLayout_4.addWidget(self.pushButton_add)
        self.pushButton_add.clicked.connect(self.add_tag)
        
        self.pushButton_del = QPushButton(self.groupBox_3)
        self.pushButton_del.setObjectName("del_btn")
        self.verticalLayout_4.addWidget(self.pushButton_del)
        self.pushButton_del.clicked.connect(self.del_tag)
        
        self.pushButton_sav = QPushButton(self.groupBox_3)
        self.pushButton_sav.setObjectName("save_tag")
        self.pushButton_sav.clicked.connect(self.on_save_tag)
        self.verticalLayout_4.addWidget(self.pushButton_sav)
        
        self.listView = QListWidget(self.groupBox_3)
        self.listView.setObjectName("tag_list_view")
        self.listView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listView.itemClicked.connect(self.on_checktag)
        
        qssStyle = '''
            QListWidget::Item {
             height :25px
            }
        '''
        self.listView.setStyleSheet(qssStyle)

        self.verticalLayout_4.addWidget(self.listView)
        self.horizontalLayout_2.addWidget(self.groupBox_3)
        self.gridLayout.addWidget(self.frame_2, 1, 0, 1, 1)
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
        self.actionOpen_Class_Folder = QAction(self)
        self.actionOpen_Class_Folder.setObjectName("actionOpen_Class_Folder")
        self.actionOpen_Class_Folder.triggered.connect(self.open_class_folder)
        self.actionLoad_Taq_List = QAction(self)
        self.actionLoad_Taq_List.setObjectName("actionLoad_Taq_List")
        self.actionLoad_Taq_List.triggered.connect(self.load_tag_list)
        self.actionSave_Taq_List = QAction(self)
        self.actionSave_Taq_List.setObjectName("actionSave_Taq_List")
        self.actionLoad_Taq_List.triggered.connect(self.on_save_tag)
        self.actionProduce_DataSets = QAction(self)
        self.actionProduce_DataSets.setObjectName("actionProduce_DataSets")
        self.actionProduce_DataSets.triggered.connect(self.on_make_dataset)
        self.actionHelp = QAction(self)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.triggered.connect(self.on_help)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Working_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Class_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Taq_List)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_Taq_List)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionProduce_DataSets)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionHelp)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi()
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("Tagor", "Tagor"))
        self.pushButton_seg.setText(_translate("Tagor", "Segman"))
        self.pushButton_sav.setText(_translate("Tagor", "Save Taq List"))
        self.pushButton_dat.setText(_translate("Tagor", "Make Train/Test Datasets"))
        self.pushButton_tag.setText(_translate("Tagor", "Tagor"))
        self.pushButton_mod.setText(_translate("Tagor", "Modler"))
        self.pushButton_sor.setText(_translate("Tagor", "Sorta"))
        self.pushButton_cpy.setText(_translate("Tagor", "Move Tagged-files to Folders"))        
        self.groupBox.setTitle(_translate("Tagor", "Working Folder"))
        self.pushButton_wor.setText(_translate("Tagor", "Open Working Folder"))
        self.groupBox_2.setTitle(_translate("Tagor", "Class Folder"))
        self.pushButton_cla.setText(_translate("Tagor", "Open Class Folder"))
        self.pushButton_las.setText(_translate("Tagor", "Last"))
        self.pushButton_nex.setText(_translate("Tagor", "Next"))        
        self.label_page.setText(_translate("Tagor", "Page Number"))
        self.groupBox_3.setTitle(_translate("Tagor", "Taq List"))
        self.pushButton_lis.setText(_translate("Tagor", "Load Taqs"))
        self.pushButton_add.setText(_translate("Tagor", "Add Taq"))
        self.pushButton_del.setText(_translate("Tagor", "Del Taq"))
        self.menuFile.setTitle(_translate("Tagor", "File"))
        self.menuHelp.setTitle(_translate("Tagor", "Help"))
        self.actionExit.setText(_translate("Tagor", "Exit"))
        self.actionOpen_Working_Folder.setText(_translate("Tagor", "Open Working Folder"))
        self.actionOpen_Class_Folder.setText(_translate("Tagor", "Open Class Folder"))
        self.actionLoad_Taq_List.setText(_translate("Tagor", "Load Taq List"))
        self.actionSave_Taq_List.setText(_translate("Tagor", "Save Taq List"))
        self.actionProduce_DataSets.setText(_translate("Tagor", "Produce DataSets"))
        self.actionHelp.setText(_translate("Tagor", "Help"))
        
    def open_work_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Load the work folder with images', '/')
        if file_path == None:
            QMessageBox.information(self,'Error','No work path selected')
        else:
            self.work_path = file_path
            self.work_paths.append(file_path)
            self.comboBox_wor.addItem(file_path.split("/")[-1])
            self.comboBox_wor.setCurrentText(file_path.split("/")[-1])
            self.image_show()

    def renew_work_folder(self):
        idx = self.comboBox_wor.currentIndex()
        self.work_path = self.work_paths[idx]
        # self.image_show()

    def open_class_folder(self):
        file_path = QFileDialog.getExistingDirectory(self, "Load Class Folder", "./")
        if file_path == None:
            QMessageBox.information(self,'Error','No class path selected')
        else:
            self.class_path = file_path
            self.class_paths.append(file_path)
            self.comboBox_cla.addItem(file_path.split("/")[-1])
            self.comboBox_cla.setCurrentText(file_path.split("/")[-1])
            self.treeView.clear()

            root = QTreeWidgetItem(self.treeView)
            root.setText(0,self.class_path.split('/')[-1])
            # root.setIcon(0,QIcon(icon))
            for _, dirs, _ in os.walk(self.class_path):
                self.CreateTree(dirs, root, self.class_path)
            QApplication.processEvents()

    def renew_class_folder(self):

        idx = self.comboBox_cla.currentIndex()
        self.class_path = self.class_paths[idx]
        self.treeView.clear()
        root = QTreeWidgetItem(self.treeView)
        root.setText(0,self.class_path.split('/')[-1])
        for _, dirs, _ in os.walk(self.class_path):
            self.CreateTree(dirs, root, self.class_path)
        QApplication.processEvents()
            
    def CreateTree(self, dirs, root, path):
        for dirname in dirs:
            path_new = path + '\\' + dirname
            if os.path.isdir(path_new):
                child = QTreeWidgetItem(root)
                child.setText(0,dirname)
                for _ , dirs_new, _ in os.walk(path_new):
                    self.CreateTree(dirs_new, child, path_new)

    def load_tag_list(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Taq List", "./", 'files(*.txt *.xml *.csv)')
        if filename == None:
            QMessageBox.information(self,'Error','Taq List load failure')
        else:
            self.tag_list_file =str(filename)
        for line in open(self.tag_list_file, "r"): #设置文件对象并读取每一行文件
            tag = line.rstrip("\n").split(',')[1]
            if tag not in tag_list:
                self.tag_list.append()
            
        self.listView.addItems(self.tag_list)
   
    def image_show(self):
        if self.work_path:
            file_path = self.work_path
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
            self.tag_ids = ["None"] * num
            self.select_ids = [False] * num
            if num < self.num_a_page:
                num_img = num
            else:
                num_img = self.num_a_page
            if num !=0:
                for i in range(num_img):
                    image_id = self.image_ids[(self.num_page - 1)*self.num_a_page + i]
                    # print(self.image_id)
                    tag_id = self.tag_ids[(self.num_page - 1)*self.num_a_page + i]
                    select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                    pixmap = QPixmap(image_id)
                    self.addImage(pixmap, image_id, tag_id, i, select_id)
                    print('load picture：', image_id)
                    QApplication.processEvents()
            else:
                QMessageBox.warning(self,'Error','Empty folder!')
                self.event(exit())
        else:
            QMessageBox.warning(self, 'Error', 'Wrong work path!')
    
    def addImage(self, pixmap, image_id, tag_id, i, select_id):

        self.col = i % 4
        self.row = i // 4

        clickable_image = QClickableImage(self.displayed_image_size, 
                                          self.displayed_image_size, pixmap, image_id, tag_id, select_id)
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
                    tag_id = self.tag_ids[(self.num_page - 1)*self.num_a_page + i]
                    select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                    pixmap = QPixmap(image_id)
                    self.addImage(pixmap, image_id, tag_id, i, select_id)
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
                tag_id = self.tag_ids[(self.num_page - 1)*self.num_a_page + i]
                select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                pixmap = QPixmap(image_id)
                self.addImage(pixmap, image_id, tag_id, i, select_id)
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
    
    def on_checktag(self):
        if self.select_ids == []:
            print("No image selected")
            self.tag_selected = self.listView.selectedItems()[0].text()
            print("Tag Selected：", (self.tag_selected))
        else:
            self.tag_selected = self.listView.selectedItems()[0].text()
            print("Tag Selected：", (self.tag_selected))
            # self.tag_ids = list("None" if x == False else self.tag_selected for x in self.select_ids)
            for i in range(len(self.tag_ids)):
                if self.select_ids[i] == True:
                    self.tag_ids[i] = self.tag_selected
            self.label_page.setText("{:d} of {:d}".format(self.num_page, (len(self.img_list)-1) // self.num_a_page+1))
            self.select_ids = [False]*len(self.tag_ids)
            if len(self.img_list) !=0:
                self.clear_layout()
                for i in range(self.num_a_page):
                    image_id = self.image_ids[(self.num_page - 1)*self.num_a_page + i]
                    # print(self.image_id)
                    tag_id = self.tag_ids[(self.num_page - 1)*self.num_a_page + i]
                    select_id = self.select_ids[(self.num_page - 1)*self.num_a_page + i]
                    pixmap = QPixmap(image_id)
                    self.addImage(pixmap, image_id, tag_id, i, select_id)
                    print('load image：', image_id)
                    QApplication.processEvents()
                    
    def on_check_class(self):
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
            self.work_path = self.class_path + "/" + path
        else:
            self.work_path = self.class_path
        self.image_show()
        print("Class selected: ", self.class_selected)

    def add_tag(self):
        # print("add new tag to tag list")
        # self.input_box = getInput()
        # self.input_box.setWindowFlags(Qt.FramelessWindowHint)
        label_hint = "Hint: use __ to seperate upper level and lower level \nA example:  Diatoms__Nitzschia__Nitzschia denticula"

        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)

        inputDialog = QInputDialog(None)
        inputDialog.setInputMode(QInputDialog.TextInput)
        inputDialog.setWindowTitle('Add New Tag')
        inputDialog.setLabelText('Enter the name for new tag:')
        inputDialog.setFont(font)
        ok = inputDialog.exec_()
        new_tag = inputDialog.textValue()

        # input_box.setWindowFlags(Qt.FramelessWindowHint)

        print('new:', new_tag)
        if ok:
            if new_tag in self.tag_list:
                QMessageBox.warning(self,'Error','taq name already in tag list')
            else:
                self.listView.addItem(str(new_tag))
                self.tag_list.append(new_tag)
        QApplication.processEvents()
        
    def del_tag(self):
        print("remove a tag selected")
        if self.tag_selected:
            msg = QMessageBox()
            confirm = msg.warning(self,'Remove a tag','Confirm to remove tag： %s'%(self.tag_selected), QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            # confirm = msg.buttonClicked.connect(self.yes_or_not)
            # print(confirm)
            if confirm == msg.Yes:
                item = self.listView.currentItem()
                self.listView.takeItem(self.listView.row(item))
                self.tag_list.remove(item.text())
        else:
            print("No tag selected")
        QApplication.processEvents()
        
       
    def on_reset(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_segman(self):
        print("uncheck all selection, and clear the tags")

    def on_start_tagor(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_modler(self):
        print("uncheck all selection, and clear the tags")
        
    def on_start_sorta(self):
        print("uncheck all selection, and clear the tags")
        
    def on_save_tag(self):
        print("save the tag list to a new file")
        if self.tag_list == []:
            QMessageBox.warning(self,'Error','tag list is empty')
        else:
            items = []
            for index in range(self.listView.count()):
                items.append(self.listView.item(index))
            labels = [i.text() for i in items]
            file_path, ok = QFileDialog.getSaveFileName(None, "save tag list to txt file", "./")
            file = open(file_path,'w')
            for i, label in enumerate(labels):
                file.write(str(i)+"," +label+ '\n')
            file.close()
            print("Tag list file produced")
        
    def on_make_dataset(self):
        # print("used the class folders to make a train dataset and test dataset")
                # if img_id:
        if self.class_path:
            self.child_window_2 = Ui_Makedataset(self.class_path)
            self.child_window_2.setWindowIcon(QIcon('./Tagor_icon.png'))
            self.child_window_2.show()
        else:
            QMessageBox.warning(self,'Error','No class path found!')
        
    def on_copy_files(self):
        print("mQMessageBox.warning(self,'Error','No image was tagged!')oving tagged images to class folder")
        if self.tag_ids == []:
            QMessageBox.warning(self,'Error','No image was tagged!')
        else:
            if not self.class_path:
                QMessageBox.warning(self,'Error','Class Folder not loaded, please load the folder')     
            else:
                for i, tag_id in enumerate(self.tag_ids):
                    if tag_id != "None":
                        tag_class = tag_id.split("__")
                        tag_path = "/".join(tag_class)
                        des_path = self.class_path + "/" + tag_path + "/"
                        if not os.path.exists(des_path):
                            os.makedirs(des_path)
                        shutil.move(self.image_ids[i], des_path)
        QMessageBox.warning(self,'Moving Finish','Images were moved to class folders with tag names')
        self.image_show()

    def on_help(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setText("Tagor \nversion 1.0.0 \n\nThank for your interest! \n\nEmail:brigchen@xmu.edu.cn \ncopyright@brigchen \nXiamen University")
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        help_box.setFont(font)
        help_box.show()
        # des_path="./"
        
#%% 每一张图片，由图片，tag名称，文件名组成；可单击和双击，双击打开图片查看器，单击选择，可多选    
class QClickableImage(QWidget):
    leftClicked = pyqtSignal()
    rightClicked = pyqtSignal()
    doubleClicked = pyqtSignal()
    def __init__(self, width = 0, height = 0, pixmap = None, image_id = '', tag_id = '', select = False):
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

        self.label_tag = QLabel()
        self.label_tag.setObjectName('label_tag')
        self.label_tag.setMinimumSize(QSize(0, 30))
        self.label_tag.setMaximumSize(QSize(width, 40))
        
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
        
        if tag_id:
            self.tag_id = tag_id
            self.label_tag.setText("Tag: {}".format(tag_id))
            self.label_tag.setAlignment(Qt.AlignCenter)
            self.label_tag.adjustSize()
            self.layout.addWidget(self.label_tag)            
                
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
    gui = Ui_Tagor()
    gui.setWindowIcon(QIcon('./Tagor_icon.png'))
    gui.showMaximized()
    sys.exit(app.exec_())  
