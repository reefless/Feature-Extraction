import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('Car4.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((10,10),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Threshold', th1)
ret,th2 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
cv2.imshow('Threshold2', th2)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations = 2)
cv2.imshow('opening', opening)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations = 2)
cv2.imshow('closing', closing)
im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im2, contours2, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
print(len(contours2))
