import cv2
import urllib.request
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
print(config["default"]["y_file"])

req = urllib.request.urlopen('http://www.studiodentaire.com/images/smile-man-open-mouth.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1) # 'Load it as it is'

cv2.imshow('lalala', img)
if cv2.waitKey() & 0xff == 27: quit()