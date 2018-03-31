import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('Car28.jpg')
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
mean = np.mean(hsv, axis=(0,1))
print(mean[0])
