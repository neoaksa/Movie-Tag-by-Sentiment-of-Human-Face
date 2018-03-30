import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.decomposition import PCA

def YtoOutput(y):
    output = np.zeros(shape=(1,8))
    output[0,int(y)] = 1.0
    return output


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

# trained face classifier
face_cascade = cv2.CascadeClassifier('/home/jie/taoj@mail.gvsu.edu/GitHub/opencv/haarcascade_frontalface_default.xml')
# cut pixel of x and y axis
resize_pixel = 200
cut_pixel_x = 30
cut_pixel_y = 40
pixel_size= (resize_pixel-2*cut_pixel_x)*(resize_pixel-2*cut_pixel_y)
x = np.empty((0,pixel_size),dtype=np.float64)
y = np.empty((0,1))
for key, item in dic_class.items():
    img_files = image_path + key[len(class_path):]
    # read img files
    img = cv2.imread(img_files)
    # find the face and resize to 200*200 pixels
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x_pix, y_pix, w, h) in faces:
        # crop the face from picture
        crop_img = gray[y_pix:y_pix + h, x_pix:x_pix + w]
    img = cv2.resize(crop_img,(resize_pixel,resize_pixel))
    # feature extraction
    # 1. SIFT
    # need to run "pip install opencv-contrib-python" if missing xfeature2d
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(crop_img,None)
    # img = cv2.drawKeypoints(img,kp,None)
    # 2. gradient
    # img = cv2.Laplacian(img, cv2.CV_64F)
    img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
    # img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # 3. canny
    # img = cv2.Canny(img, 100, 200)
    # cut pixel
    img = img[cut_pixel_x:200-cut_pixel_x,cut_pixel_y:200-cut_pixel_y]
    # PCA
    # pca = PCA(n_components=6, whiten=True,svd_solver='randomized',)
    # pca.fit(img)
    # pca_s = pca.components_[0:3]
    # plot face
    cv2.imshow(dic_class[key],img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # normalization
    img = img / 255
    print(len(img.flatten()))
    x = np.vstack((x, img.flatten()))
    y = np.vstack((y, dic_class[key]))

cv2.destroyAllWindows()
y_tag = [YtoOutput(item) for item in y]
y_tag = np.vstack(y_tag)
np.save("/home/jie/Documents/x.npy",x)
np.save("/home/jie/Documents/y.npy",y_tag)
print("saved!")

x = np.load("/home/jie/Documents/x.npy")
y = np.load("/home/jie/Documents/y.npy")
# split training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
clf = MLPClassifier(hidden_layer_sizes=(200,100), max_iter=50,learning_rate="adaptive",
                    learning_rate_init=0.3,momentum=0.5,activation="logistic",
                     solver='sgd', verbose=True,  random_state=10, batch_size=10)

clf.fit(x_train, y_train)
joblib.dump(clf, '/home/jie/Documents/clf.pkl')

# clf = joblib.load('/home/jie/Documents/clf.pkl')
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


