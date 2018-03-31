import cv2
import numpy as np
img = cv2.imread('Car34.jpg')
img = np.float32(img) / 255.0
 
# Calculate gradient 
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
#print(mag)
print (np.mean(mag))
print(np.mean(angle))
