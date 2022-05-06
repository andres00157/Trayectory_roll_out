# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 07:23:21 2021

@author: xboxk
"""


import cv2
import numpy as np

img= cv2.imread("mapa/mapa_laberinto.png")

output = cv2.resize(img, [img.shape[1]*2,img.shape[0]*2])

cv2.imwrite("mapa/mapa_laberinto_2.png",output)

cv2.imshow("img",img)
cv2.imshow("img_2",output)
cv2.waitKey(0)