import cv2
import urllib.request
import numpy as np
from numpy import genfromtxt
import configparser
import random

#
# config = configparser.ConfigParser()
# config.read("config.ini")
# print(config["default"]["y_file"])
# print(random.sample(set([5,-5]),1))
# req = urllib.request.urlopen('http://www.studiodentaire.com/images/smile-man-open-mouth.jpg')
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# img = cv2.imdecode(arr, -1) # 'Load it as it is'
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# row, col= img.shape
# roationmarix = cv2.getRotationMatrix2D((col/2,row/2),random.sample(set([5,-5]),1)[0],1)
# img2 = cv2.warpAffine(img, roationmarix, (col,row))
# cv2.imshow('lalala', img)
# cv2.imshow('laldala', img2)
# if cv2.waitKey() & 0xff == 27: quit()

imgarr = genfromtxt("/media/d/human face/fer2013/fer2013.csv", delimiter=",")
print(imgarr.shape)