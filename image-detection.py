from sklearn.decomposition import PCA
import numpy as np
import cv2
import dlib
from sklearn.externals import joblib
import urllib.request
import configparser
import face_enum

# 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise)
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)

    # return the image
    return image

# read config file
config = configparser.ConfigParser()
config.read("config.ini")
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(config["default"]["face_cascade"])
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier('/home/jie/taoj@mail.gvsu.edu/GitHub/opencv/haarcascade_eye.xml')


filename = config["web_face"]["face_link"]
# img = url_to_image(filename)

filename = config["web_face"]["face_local"]
img = cv2.imread(filename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
pca = joblib.load(config["default"]["pca_model"])
# clf = joblib.load(config["default"]["mlp_model"])
clf = joblib.load(config["default"]["svm_model"])
dlib_db = config["default"]["dlib_db"]
# cut pixel of x and y axis
resize_pixel = 200
cut_pixel_x = int(resize_pixel * 0)
cut_pixel_y = int(resize_pixel * 0)
pixel_size= resize_pixel * resize_pixel
for (x_pix, y_pix, w, h) in faces:
    # crop the face from picture
    cv2.rectangle(img, (x_pix, y_pix), (x_pix + w, y_pix + h), (255, 0, 0), 2)
    # detect
    # crop the face from picture, move lower to reduce the top of head
    crop_img = gray[y_pix + cut_pixel_y:y_pix - cut_pixel_y + h, x_pix + cut_pixel_x:x_pix - cut_pixel_x + w]

# resize
crop_img = cv2.resize(crop_img,(resize_pixel,resize_pixel))
predictor = dlib.shape_predictor(dlib_db)
rect = dlib.rectangle(0, 0, resize_pixel, resize_pixel)
landmarks = np.matrix([[p.x, p.y] for p in predictor(crop_img, rect).parts()])
landmarks = np.squeeze(np.array(landmarks))
FACE_POINTS = [
    # list(range(0, 17)),  # JAWLINE_POINTS
    list(range(17, 22)),  # RIGHT_EYEBROW_POINTS
    list(range(22, 27)),  # LEFT_EYEBROW_POINTS
    # list(range(27, 36)),  # NOSE_POINTS
    list(range(36, 42)),  # RIGHT_EYE_POINTs
    list(range(42, 48)),  # LEFT_EYE_POINTS
    list(range(48, 61)),  # MOUTH_OUTLINE_POINTS
    list(range(61, 68))]  # MOUTH_INNER_POINTS
# link related points
for face_lists in FACE_POINTS:
    for point in face_lists[1:]:
        cv2.line(crop_img, (landmarks[point - 1][0], landmarks[point - 1][1]),
                 (landmarks[point][0], landmarks[point][1]), (0, 0, 255), 10)
crop_img = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0, ksize=5)  # gradient by x axis
show_img = crop_img
min, max = crop_img.min(), crop_img.max()
crop_img = (crop_img - min)/(max-min)
crop_img = crop_img.flatten()
x = pca.transform(crop_img.reshape(1,crop_img.shape[0]))
y_pred = clf.predict(x)
# 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise)
print(y_pred)
# print(y_pred.argmax(axis=1))

# put result into image
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,int(img.shape[0]*0.9))
fontScale = 1
fontColor = (0,0,255)
lineType = cv2.LINE_4


cv2.putText(img, face_enum.Face(y_pred[0]).name, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
# eyes = eye_cascade.detectMultiScale(roi_gray)
# for (ex, ey, ew, eh) in eyes:
#     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.imshow('crop_img', show_img)
cv2.moveWindow("crop_img",600,200)
if cv2.waitKey() & 0xff == 27:
    quit()

