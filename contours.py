import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('Building15.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
print(len(contours))
M = cv2.moments(cnt)
print(M)
x,y,w,h = cv2.boundingRect(cnt)
print(x,y,w,h)
kernel = np.ones((10,10),np.uint8)
erosion = cv2.erode(image,kernel,iterations = 1)

img = cv2.rectangle(im,(x,y),(x+10,y+10),(0,255,0),2)

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
kernel2 = np.ones((10,10),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
ii, cc, hh=cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(cc))
cv2.imshow('img',img)
cv2.imshow('Opening',opening)
cv2.imshow('erosion',erosion)
cv2.imshow('closing',closing)
cv2.imwrite('blob.jpg', closing)
plt.subplot(131),plt.imshow(im, cmap= None)
plt.subplot(132),plt.imshow(thresh, cmap= None)
plt.subplot(133),plt.imshow(image, cmap= None)
plt.show()
