from sklearn.decomposition import PCA
import numpy as np
import cv2
import dlib
from sklearn.externals import joblib

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('/home/jie/taoj@mail.gvsu.edu/GitHub/opencv/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('/home/jie/taoj@mail.gvsu.edu/GitHub/opencv/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    pca = joblib.load('/home/jie/Documents/pca.pkl')
    clf = joblib.load('/home/jie/Documents/clf.pkl')
    for (x, y, w, h) in faces:
        # crop the face from picture
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # detect
        dlib_db = "./dlib/shape_predictor_68_face_landmarks.dat"
        crop_img = gray[y:y+h,x:x+w]
        # predictor = dlib.shape_predictor(dlib_db)
        # rect = dlib.rectangle(0, 0, h, w)
        # landmarks = np.matrix([[p.x, p.y] for p in predictor(crop_img, rect).parts()])
        # landmarks = np.squeeze(np.array(landmarks))
        # FACE_POINTS = [
        #     # list(range(0, 17)),  # JAWLINE_POINTS
        #     list(range(17, 22)),  # RIGHT_EYEBROW_POINTS
        #     list(range(22, 27)),  # LEFT_EYEBROW_POINTS
        #     # list(range(27, 36)),  # NOSE_POINTS
        #     list(range(36, 42)),  # RIGHT_EYE_POINTs
        #     list(range(42, 48)),  # LEFT_EYE_POINTS
        #     list(range(48, 61)),  # MOUTH_OUTLINE_POINTS
        #     list(range(61, 68))]  # MOUTH_INNER_POINTS
        # # link related points
        # for face_lists in FACE_POINTS:
        #     for point in face_lists[1:]:
        #         cv2.line(crop_img, (landmarks[point - 1][0], landmarks[point - 1][1]),
        #                  (landmarks[point][0], landmarks[point][1]), (0, 0, 255), 10)
        crop_img = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0, ksize=5)  # gradient by x axis
        crop_img = cv2.resize(crop_img,(200,200))
        crop_img *= 255.0 / crop_img.max()
        crop_img = crop_img.flatten()
        x = pca.transform(crop_img.reshape(1,crop_img.shape[0]))
        y_pred = clf.predict(x)
        # (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        print(y_pred.argmax(axis=1))
        #
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('crop_img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()