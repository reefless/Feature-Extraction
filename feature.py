import cv2
import numpy as np
import glob
import pandas as pd
import sys
from PyQt5 import QtGui, uic, QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from os.path import expanduser
########Global Variable########
MIN_MATCH_COUNT = 20
carTemplate = cv2.imread('Carxxx.jpg',0)
lower_blue = np.array([30,30,20])
upper_blue = np.array([90,255,255])
hsvPercentage = []
path = ''
imgname = ''
img = ''
###############################
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('CSV_UI.ui', self)
        self.openBtn.clicked.connect(self.get_path)
        self.exportBtn.clicked.connect(self.findFeatures)
        self.ExitBtn.clicked.connect(self.exit_app)
        self.show()
    def get_path(self):
        global path
        path = QFileDialog.getExistingDirectory(self, "Open a folder", expanduser(''), QFileDialog.ShowDirsOnly)
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
    def find_min(self):
        global img
        mina = np.amin(img)
        return mina
    def find_max(self):
        global img
        maxa = np.amax(img)
        return maxa
    def find_omean(self):
        global img
        mean = np.mean(img)
        return mean
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
                #print('Mask[y,x] = ')
                #print(mask[y,x])
                if mask[y,x] == 255:
                    count = count + 1
        return (count/float(overall))*100
    def find_SURF(img):
        surf = cv2.xfeatures2d.SURF_create(1000)
        kp, des = surf.detectAndCompute(img,None)
        return len(kp)

    def find_SIFT(imgSift):
        #Find SIFT Keypoint Column
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(carTemplate,None)
        kp2, des2 = sift.detectAndCompute(imgSift,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            hS,wS = carTemplate.shape
            pts = np.float32([ [0,0],[0,hS-1],[wS-1,hS-1],[wS-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            imgSift = cv2.polylines(imgSift,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print("Not enough matches are found")
            #print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(carTemplate,kp1,imgSift,kp2,good,None,**draw_params)
        if matchesMask != None:
            SIFT_count = len(matchesMask)
            #print(len(matchesMask))
        else:
            SIFT_count = 0
        return SIFT_count

    def find_Canny(img):
        edges = cv2.Canny(img,200,800)
        cannyE = np.sum(edges)
        return cannyE
    def find_OTSU(self):
        global img
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((10,10),np.uint8)
        kernel2 = np.ones((5,5),np.uint8)
        img = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations = 2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations = 2)
        im2, contours2, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return len(contours2)
    def findFeatures(self):
        global hsvPercentage
        global path
        count = 0
        for file in glob.glob(path + '/*.jpg'):
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
            #Find Min
            #mina = self.find_min()
            #Find Max
            #maxa = self.find_max()
            
            #Find HSV Percentage Column
            hsv_perc = self.find_HSV()
            #Find Contours
            contours = self.find_contour()
            #Find Overall Mean
            omean = self.find_omean()
            #Find SURF
            #lenKP = find_SURF(img)

            #Find SIFT
            #SIFT_count = find_SIFT(imgSift)
                
            #Find Canny Edge
            #cannyE = find_Canny(img)
            #Find OTSU
            otsu = self.find_OTSU()
            #Insert Value    
            value = (hsv_perc,mean[0],mean[1],mean[2],omean,contours,otsu,classname)
            #value = (hsv_perc,mean[0],otsu,classname)
            #value = (hsv_perc,mean[1], omean,contours,classname)
            hsvPercentage.append(value)
            count = count + 1
        #Push value to DataFrame
        #column_name = ['HSV Percentage','B Mean','OTSU','Class Name']
        #column_name = ['HSV', 'B Mean', 'Overall Mean', 'contours','ClassName']
        column_name = ['HSV Percentage','B Mean','G Mean', 'R Mean','Overall Mean','contours','OTSU','Class Name']
        df = pd.DataFrame(hsvPercentage, columns=column_name)
        df.to_csv('withoutUU2.csv', index=None)

        #print(hsvPercentage)
        print(count)
        print("CSV file created!")
        QMessageBox.about(self, "CSV", "CSV file created!")
    def exit_app(self):
        sys.exit()
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())

