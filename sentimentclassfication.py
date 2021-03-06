import os
import cv2
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import math
import dlib
import matplotlib.pyplot as plt
import configparser

# --------------------------------
# self defined functions area
def YtoOutput(y):
    output = np.zeros(shape=(1,7))
    output[0,int(y)] = 1.0
    return output[0]


# oversample
def oversample(img):
    seed = random.randint(1,4)
    # horitional flip
    if seed == 1:
        img = cv2.flip(img,1)
    # brintness or darkness
    elif seed == 2:
        img = img -10
    # rotate +-5 degree
    elif seed == 3:
        row, col = img.shape
        rotatematrix = cv2.getRotationMatrix2D((col / 2, row / 2),
                                               random.sample(set([3,-3]),1)[0], 1)
        img = cv2.warpAffine(img, rotatematrix, (col, row))
    return img

# --------------------------------
# read config file
config = configparser.ConfigParser()
config.read("config.ini")
# read the file of image and classification
image_path = config["default"]["image_path"]
class_path = config["default"]["class_path"]

# # read motion files
# # dictionary for saving classification corresponding image path
# dic_class = {}
# for root, dictionaries, files in os.walk(class_path):
#     for file in files:
#         path = os.path.join(root,file)
#         with open(path, "r", encoding="utf-8") as newfile:
#             txtdata = newfile.read()
#             # set the classification to dictionary
#             dic_class[path[:-12]+".png"] = str(int(txtdata[3]) - 1)
#             # more extended sample(optional)
#             path_pre1 = path[:-20] \
#                         + (8-len(str(int(path[-20:-12])-1)))*"0" \
#                         + str(int(path[-20:-12])-1)
#             path_pre2 = path[:-20] \
#                         + (8 - len(str(int(path[-20:-12]) - 2))) * "0" \
#                         + str(int(path[-20:-12]) - 2)
#             dic_class[path_pre1 + ".png"] = str(int(txtdata[3]) - 1)
#             dic_class[path_pre2 + ".png"] = str(int(txtdata[3]) - 1)
#
# # trained face classifier
# face_cascade = cv2.CascadeClassifier(config["default"]["face_cascade"])
# dlib_db = config["default"]["dlib_db"]
# # cut pixel of x and y axis
# resize_pixel = 200
# cut_pixel_x = int(resize_pixel * 0.15)
# cut_pixel_y = int(resize_pixel * 0.15)
# # SIFT has different matrix size
# SIFT = False
# n_kmean = 50
# if SIFT:
#     pixel_size = n_kmean*128
# else:
#     pixel_size = resize_pixel * resize_pixel
# x = np.empty((0, pixel_size), dtype=np.float16)
# y = np.empty((0,1))
# for key, item in dic_class.items():
#     img_files = image_path + key[len(class_path):]
#     # read img files
#     print(img_files, "  ",dic_class[key])
#
#     img = cv2.imread(img_files)
#     # find the face and resize to 200*200 pixels
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # random disturbance
#     gray = oversample(gray)
#     try:
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         for (x_pix, y_pix, w, h) in faces:
#             # crop the face from picture, move lower to reduce the top of head
#             crop_img = gray[y_pix + cut_pixel_y:y_pix - cut_pixel_y + h, x_pix + cut_pixel_x:x_pix - cut_pixel_x + w]
#         img = cv2.resize(crop_img,(resize_pixel,resize_pixel))
#         # --------------------------------
#         # feature extraction
#         # 4. Dlib， Dlib need to be used with other function
#         if not SIFT:
#             predictor = dlib.shape_predictor(dlib_db)
#             rect = dlib.rectangle(0, 0, resize_pixel, resize_pixel)
#             landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
#             landmarks = np.squeeze(np.array(landmarks))
#             FACE_POINTS = [
#                 # list(range(0, 17)),  # JAWLINE_POINTS
#                 list(range(17, 22)),  # RIGHT_EYEBROW_POINTS
#                 list(range(22, 27)),  # LEFT_EYEBROW_POINTS
#                 # list(range(27, 36)),  # NOSE_POINTS
#                 list(range(36, 42)),  # RIGHT_EYE_POINTs
#                 list(range(42, 48)),  # LEFT_EYE_POINTS
#                 list(range(48, 61))]  # MOUTH_OUTLINE_POINTS
#                 # list(range(61, 68))]  # MOUTH_INNER_POINTS
#             # link related points
#             for face_lists in FACE_POINTS:
#                 for point in face_lists[1:]:
#                     cv2.line(img, (landmarks[point - 1][0], landmarks[point - 1][1]),
#                              (landmarks[point][0], landmarks[point][1]), (0, 0, 255), 5)
#         # 1. SIFT
#         # need to run "pip install opencv-contrib-python" if missing xfeature2d
#         sift = cv2.xfeatures2d.SIFT_create()
#         kp,kpdesc = sift.detectAndCompute(crop_img,None)
#         # k-mean to group keypoint
#         img = KMeans(n_clusters=n_kmean,random_state=0).fit(kpdesc).cluster_centers_
#         # img = cv2.drawKeypoints(img,kp,None)
#         # 2. gradient
#         # img = cv2.Laplacian(img, cv2.CV_64F)  # Laplacian
#         # img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)   # gradient by x axis
#         # img = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)   # gradient by y axis
#         # img_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5) # gradient by degree
#         # img_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5) # gradient by degree
#         # img = cv2.phase(img_x,img_y,angleInDegrees=True, angle=0.45 )    # gradient by degree
#         # 3. canny
#         # img = cv2.Canny(img, 50, 200)
#         # plot face
#         cv2.imshow(dic_class[key],img)
#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break
#         # normalization
#         min, max = img.min(), img.max()
#         img = (img - min)/(max-min)
#         # print(len(img.flatten()))
#         x = np.vstack((x, img.flatten()))
#         y = np.vstack((y, dic_class[key]))
#     except:
#         continue
#
# # --------------------------------
# # save pre-handled data
# cv2.destroyAllWindows()
# y_tag = [YtoOutput(item) for item in y]
# np.save(config["default"]["x_file"],x)
# np.save(config["default"]["y_file"],y_tag)
# print("saved!")

# --------------------------------
# training the dataset
x = np.load(config["default"]["x_file"])
y = np.load(config["default"]["y_file"])
# --------------------------------
# PCA analysis to reduce features
pca = PCA(n_components=100, whiten=True)
x = pca.fit_transform(x)
joblib.dump(pca, config["default"]["pca_model"])
# plot PCA curve
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()
# --------------------------------
# split training and validation dataset
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=10)
for train_index, test_index in sss.split(x,y):
    x_train, x_test = x[train_index,:], x[test_index,:]
    y_train, y_test = y[train_index,:], y[test_index,:]
# --------------------------------
# MLP model
# clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500,learning_rate="adaptive",
#                     learning_rate_init=0.1,momentum=0.5,activation="logistic",
#                      solver='sgd', verbose=True,  random_state=10, batch_size=10)
# clf.fit(x_train, y_train)
# # save model
# joblib.dump(clf, config["default"]["mlp_model"])
# clf = joblib.load(config["default"]["mlp_model"])
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
# print(f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1), average=None))
# ------------------------------------
# SVM model
clf = svm.SVC(kernel="rbf",C=5,verbose=True)
clf.fit(x_train, y_train.argmax(axis=1))
joblib.dump(clf, config["default"]["svm_model"])
# load model and validation
# clf = joblib.load(config["default"]["svm_model"])
y_pred = clf.predict(x_test)
print(accuracy_score(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1),y_pred))
print(f1_score(y_test.argmax(axis=1),y_pred, average=None))




