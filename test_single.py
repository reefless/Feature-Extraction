import cv2
import numpy as np

img = cv2.imread('Tree11.jpg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#print(hsv)
cv2.imshow('Original Image', img)
cv2.imshow('HSV Image', hsv)
lower_blue = np.array([30,30,20])
upper_blue = np.array([90,255,255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)
#print(mask)

h = mask.shape[0]
w = mask.shape[1]
count = 0
overall = h*w
lower_blue = np.array([30,30,30])
upper_blue = np.array([90,255,255])
for y in range(0,h):
        for x in range(0,w):
            if mask[y,x] == 255:
                count = count + 1

print((count/overall)*100)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
