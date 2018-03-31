import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Car30.jpg', 0)
edges = cv2.Canny(img,200,800)

cv2.imshow('Original Image', img)
cv2.imshow('Canny', edges)
print(np.sum(edges))
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
