from sklearn import datasets, svm, neighbors, ensemble,model_selection, tree, multiclass
from sklearn.externals import joblib

import cv2
import numpy as np
from numpy import genfromtxt
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QFileDialog, QLabel, QMessageBox 
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from os.path import expanduser

######## STITCH ########
ptB = []
ptD = []
ptF = []
ptE = []
max_ptB = []
max_ptD = []
max_ptF = []
max_ptE = []
images = []
image = []
resultimage = []
path = ''
########################

######## TRAIN ########
path2 = ''
hsvPercentage = []
imgname = ''
img = ''
MIN_MATCH_COUNT = 20
lower_blue = np.array([30,30,20])
upper_blue = np.array([90,255,255])

########################




class App(QMainWindow):

    def __init__(self):
        super(QWidget,self).__init__()
        self.title = 'Image Segmentation'
        self.left = 0
        self.top = 0
        self.width = 400
        self.height = 300
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.show()

class MyTableWidget(QWidget):        

    def __init__(self, parent):   
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()	
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(400,300) 

        # Add tabs
        self.tabs.addTab(self.tab1,"Train")
        self.tabs.addTab(self.tab2,"Stitch")
        self.tabs.addTab(self.tab3,"Test")


        # Create first tab
        self.tab1.layout = QVBoxLayout(self)
        self.pushButton1 = QPushButton("Open Folder")
        self.pushButton1.clicked.connect(self.get_path)
        self.pushButton2 = QPushButton("Generate CSV file")
        self.pushButton2.clicked.connect(self.called_train)
        self.pushButton3 = QPushButton("Exit")
        self.pushButton3.clicked.connect(self.exit_app)
        self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.layout.addWidget(self.pushButton2)
        self.tab1.layout.addWidget(self.pushButton3)

        self.tab1.setLayout(self.tab1.layout)

         # Create Second tab
        self.tab2.layout = QVBoxLayout(self)
        self.pushButton4 = QPushButton("Open Folder")
        self.pushButton4.clicked.connect(self.open_file)
        self.pushButton5 = QPushButton("Stitch image")
        self.pushButton5.clicked.connect(self.read_file)
        self.pushButton6 = QPushButton("Save Image")
        self.pushButton6.clicked.connect(self.save_file)
        self.pushButton7 = QPushButton("Exit")
        self.pushButton7.clicked.connect(self.exit_app)
        self.tab2.layout.addWidget(self.pushButton4)
        self.tab2.layout.addWidget(self.pushButton5)
        self.tab2.layout.addWidget(self.pushButton6)
        self.tab2.layout.addWidget(self.pushButton7)
        self.tab2.setLayout(self.tab2.layout)
        
         # Create Third tab
        self.tab3.layout = QVBoxLayout(self)
        self.pushButton8 = QPushButton("Open Test Pictures")
        self.pushButton9 = QPushButton("Exit")
        self.tab3.layout.addWidget(self.pushButton8)
        self.tab3.layout.addWidget(self.pushButton9)
        self.tab3.setLayout(self.tab3.layout)



        # Add tabs to widget        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

######################### STITCH #########################
        
    def open_file(self):
        my_dir = QFileDialog.getExistingDirectory(self,"Open a folder", expanduser(''), QFileDialog.ShowDirsOnly)
        global path
        path = my_dir

    def exit_app(self):
        sys.exit()

    def read_file(self):
        i = 0
        for data in sorted(glob.glob(path + '/*.JPG')):
            global image
            image.append(cv2.imread(data,0))
            i=i+1
            global images
            images.append(cv2.imread(data))
        self.stitch()
        self.stitch2()
        
    def stitch(self):

        image1 = images[0]
        image2 = images[1]
        image1 = cv2.resize(image1, (1000, 750))
        image2 = cv2.resize(image2, (1000, 750))

        (kpsA, featuresA) = self.findKeyPoints(image1)
        (kpsB, featuresB) = self.findKeyPoints(image2)
        M = self.matchedKeyPoints(kpsA, kpsB, featuresA, featuresB)

        if M is None:
            return None
        
        (matches, status, H) = M

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptB.append(kpsB[trainIdx][1])
        
        max_ptB = sorted(ptB, reverse=True)
        sum_ptB = 0
        for i in range(9): 
            sum_ptB = sum_ptB + max_ptB[i]
        avg_ptB = sum_ptB/10
        avg_ptB = int(avg_ptB)

        cropImg = image2[:,avg_ptB:]

        connect = np.hstack((image1, cropImg))
        cv2.imwrite("Result/stitch1.jpg", connect)
         
    def stitch2(self):
        image3 = images[2]
        image4 = images[3]
        image3 = cv2.resize(image3, (1000, 750))
        image4 = cv2.resize(image4, (1000, 750))
        
        (kpsC, featuresC) = self.findKeyPoints(image3)
        (kpsD, featuresD) = self.findKeyPoints(image4)
        M = self.matchedKeyPoints(kpsC, kpsD, featuresC, featuresD)
        (matches, status, H) = M
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptD.append(kpsD[trainIdx][1])
        
        max_ptD = sorted(ptD, reverse=True)
        sum_ptD = 0
        for i in range(9): 
            sum_ptD = sum_ptD + max_ptD[i]
        avg_ptD = sum_ptD/10
        avg_ptD = int(avg_ptD)

        cropImg2 = image4[:,avg_ptD:]
        connect2 = np.hstack((image3, cropImg2))
        cv2.imwrite("Result/stitch2.jpg", connect2)

        self.stitch3()
        
    def stitch3(self):
        image5 = cv2.imread('Result/stitch1.jpg')
        image6 = cv2.imread('Result/stitch2.jpg')
        image5 = cv2.resize(image5, (2000,1500))
        image6 = cv2.resize(image6, (2000,1500))
        (kpsE, featuresE) = self.findKeyPoints(image5)
        (kpsF, featuresF) = self.findKeyPoints(image6)
        M = self.matchedKeyPoints(kpsE, kpsF, featuresE, featuresF)
        (matches, status, H) = M
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptF.append(kpsF[trainIdx][0])
        
        max_ptF = sorted(ptF, reverse=True)
        sum_ptF = 0
        for i in range(9): 
            sum_ptF = sum_ptF + max_ptF[i]
        avg_ptF = sum_ptF/10
        avg_ptF = int(avg_ptF)

        cropImg3 = image6[:,avg_ptF:]
        connect3 = np.hstack((image5, cropImg3))
        connect3 = cv2.resize(connect3, (4000, 1500))
        global resultimage
        resultimage = connect3
        
        cv2.imshow("finalstitch", connect3)
          
    def findKeyPoints(self, image):

        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def matchedKeyPoints(self, kpsA, kpsB, featuresA, featuresB):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
               
            return (matches, status, H)
        return None

    def save_file(self):
 
        cv2.imwrite('Result/stitchimage.jpg', resultimage)
        QMessageBox.about(self, "File", "Saved File")
            
    
    ##########################################################
    ######################### TRAIN #########################


    def get_path(self):
        global path2
        path2 = QFileDialog.getExistingDirectory(self, "Open a folder", expanduser(''), QFileDialog.ShowDirsOnly)

    def findFeatures(self):
        global hsvPercentage
        global path2
        count = 0
        for file in glob.glob(path2 + '/*.jpg'):
            SIFT_count = 0
            global img
            img = cv2.imread(file, 1)
            imgSift = cv2.imread(file,0)
            print(file)
            global imgname
            param,imgname = file.split("/",1)
            print(imgname)
            
            #Find Class Name Column
            classname = self.find_classname()
            #Find Mean
            mean = self.find_mean()           
            #Find HSV Percentage Column
            hsv_perc = self.find_HSV()
            #Find Contours
            contours = self.find_contour()
            #Find Overall Mean
            omean = self.find_omean()          
            #Find OTSU
            otsu = self.find_OTSU()
            #Insert Value    
            value = (hsv_perc,mean[0],mean[1],mean[2],omean,contours,otsu,classname)

            hsvPercentage.append(value)
            count = count + 1
        #Push value to DataFrame

        column_name = ['HSV Percentage','B Mean','G Mean', 'R Mean','Overall Mean','contours','OTSU','Class Name']
        df = pd.DataFrame(hsvPercentage, columns=column_name)
        df.to_csv('data.csv', index=None)

        #print(hsvPercentage)
        print(count)
        print("CSV file created!")
        QMessageBox.about(self, "CSV", "CSV file created!")

    def find_classname(self):
        global imgname
        classname = ""
        if "Tree" in imgname:
            classname = "Green Zone"
        if "Building" in imgname:
            classname = "Building"
        if "Car" in imgname:
            classname = "Car"
        if "Road" in imgname:
            classname = "Road"
        if "Canal" in imgname:
            classname = "Canal"
        if "Unknown" in imgname:
            classname = "Unknown"
        return classname
    
    def find_mean(self):
        global img
        mean = np.mean(img, axis=(0,1))
        return mean
    
    def find_HSV(self):
        global img
        count = 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        h = mask.shape[0]
        w = mask.shape[1]
        overall = h*w
        for y in range(0,h):
            for x in range(0,w):
                if mask[y,x] == 255:
                    count = count + 1
        return (count/float(overall))*100
    
    def find_contour(self):
        global img 
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((30,30),np.uint8)
        kernel2 = np.ones((70,70),np.uint8)
        img = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        return len(contours)

    def find_omean(self):
        global img
        mean = np.mean(img)
        return mean

    def find_OTSU(self):
        global img
        kernel = np.ones((10,10),np.uint8)
        kernel2 = np.ones((5,5),np.uint8)
        img = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations = 2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations = 2)
        im2, contours2, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return len(contours2)

    def called_train(self):
        self.findFeatures()        

    ##########################################################


    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())





