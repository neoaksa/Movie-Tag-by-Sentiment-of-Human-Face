import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def YtoOutput(y):
    output = np.zeros(shape=(1,8))
    output[0,int(y)] = 1.0
    return output

'''
# read the file of image and classification
image_path = "/media/d/human face/cohn-kanade-images/"
class_path = "/home/jie/Documents/Emotion/"

# read motion files
# dictionary for saving classification corresponding image path
dic_class = {}
for root, dictionaries, files in os.walk(class_path):
    for file in files:
        path = os.path.join(root,file)
        with open(path, "r", encoding="utf-8") as newfile:
            txtdata = newfile.read()
            # set the classification to dictionary
            dic_class[path[:-12]+".png"] = txtdata[3]

x = np.empty((0,129500),dtype=np.float64)
y = np.empty((0,1))
for key, item in dic_class.items():
    img_files = image_path + key[len(class_path):]
    # read img files
    img = cv2.imread(img_files)
    # crop the edges, origal : 490 * 640
    img = img[30:400, 150:500]
    # transfer to gray which is numpy array
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img,100,200)
    # plot face
    cv2.imshow(dic_class[key],img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # normalization
    img = img / 255
    # print(len(img.flatten()))
    x = np.vstack((x, img.flatten()))
    y = np.vstack((y, dic_class[key]))

y_tag = [YtoOutput(item) for item in y]
y_tag = np.vstack(y_tag)
np.save("/home/jie/Documents/x.npy",x)
np.save("/home/jie/Documents/y.npy",y_tag)
'''

x = np.load("/home/jie/Documents/x.npy")
y = np.load("/home/jie/Documents/y.npy")
# split training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=100,learning_rate="adaptive",
                    learning_rate_init=0.1,momentum=0.9,activation="logistic",
                     solver='adam', verbose=True,  random_state=10, batch_size=10)

clf.fit(x_train, y_train)
joblib.dump(clf, '/home/jie/Documents/clf.pkl')

# clf = joblib.load('/home/jie/Documents/clf.pkl')
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


