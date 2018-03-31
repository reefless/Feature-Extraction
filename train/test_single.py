from sklearn import datasets, svm, neighbors, ensemble,model_selection, tree, multiclass
import numpy as np
from numpy import genfromtxt

from sklearn.externals import joblib
import cv2

alpha = 0.2
imgset = cv2.imread('test_data/Road/Road678.jpg') 
stride = 200
interval = 200
testData = []
h= imgset.shape[0]
print(h)
w=imgset.shape[1]
print(w)
ar = []
lower_blue = np.array([30,30,20],np.uint8)
upper_blue = np.array([90,255,255],np.uint8)

def find_mean(img):
    mean = np.mean(img, axis=(0,1))
    return mean
def find_min(img):
    mina = np.amin(img)
    return mina
def find_max(img):
    maxa = np.amax(img)
    return maxa
def find_omean(img):
    mean = np.mean(img)
    return mean
def find_contour(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((30,30),np.uint8)
    kernel2 = np.ones((70,70),np.uint8)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)
def find_HSV(img):
    count = 0
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower_blue, upper_blue)
    height = mask.shape[0]
    width = mask.shape[1]
    overall = height*width
    for y in range(0,height):
        for x in range(0,width):
            if mask[y,x] == 255:
                count = count + 1
    return (count/float(overall))*100
def find_OTSU(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((10,10),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations = 2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations = 2)
    im2, contours2, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours2)
    
def callFeatures(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv = find_HSV(hsv)
    mean = find_mean(im)
    omean = find_omean(im)
    contours = find_contour(im)
    otsu = find_OTSU(im)
    value = (hsv, mean[0], mean[1], mean[2], omean, contours, otsu)
    testData.append(value)
callFeatures(imgset)
clf = joblib.load('model_withoutUU2_extra.pkl')
prediction = clf.predict(testData)
print("Extra" + str(prediction))

clf = joblib.load('model_withoutUU2_KNN.pkl')
prediction = clf.predict(testData)
print("KNN"+ str(prediction))

clf = joblib.load('model_withoutUU2_Decision.pkl')
prediction = clf.predict(testData)
print("Decision"+ str(prediction))

cv2.imshow("output",imgset)
k = cv2.waitKey(0)
if k==27:    # Esc key to stop
    cv2.destroyAllWindows()

