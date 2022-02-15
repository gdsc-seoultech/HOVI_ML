import cv2
import dlib
from imutils import face_utils, resize 
import numpy as np #행렬 처리

orange_img = cv2.imread('orange.jpg')
# orange_img = cv2.resize(orange_img, dsize=(512, 512))

detector = dlib.get_frontal_face_detector() #얼굴 영역 탐지
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0) #웹캠 사용!

while cap.isOpened():
    ret, img = cap.read() #이미지 읽어오기

    if not ret:
        break

    faces = detector(img) #좌표 저장

    if len(faces) > 0:
        face = faces[0] #face가 여러 개일 때는 idx=0인 얼굴만 사용
    else:
        face = faces

    shape = predictor(img, face)
    shape = face_utils.shape_to_np(shape)

    #img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    
    for s in shape:
        cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey(1)