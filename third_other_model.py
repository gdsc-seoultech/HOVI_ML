import cv2
import dlib
from imutils import face_utils, resize 
import numpy as np #행렬 처리

scaler = 0.3

detector = dlib.get_frontal_face_detector() #얼굴 영역 탐지
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmark.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read() #이미지 읽어오기

    if not ret:
        break
    
    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    img = resize(img, width = 1000)

    faces = detector(img) #좌표 저장

    if len(faces) > 0:
        face = faces[0] #face가 여러 개일 때는 idx=0인 얼굴만 사용

        shape = predictor(img, face)
        shape = face_utils.shape_to_np(shape)

        img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
    
        for s in shape:
            cv2.circle(img, center=tuple(s), radius=1, color=255, thickness=1, lineType=cv2.LINE_AA)
        
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1) #역시 마진

        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)
        
        ##############################  의  사  표  현  ##############################
        # 측정치

        # 입 벌리기
        mouth_y1 = shape[62,1]
        mouth_y2 = shape[66,1]
        # print(mouth_y1-mouth_y2)

        # 웃기
        print(shape[54,0] - shape[48,0]) 
        
        #오른쪽 눈 감기 - 실패: 랜드마크들이 그 자리에 고정되어 있음
        le_y1 = shape[37, 1]
        le_y2 = shape[41, 1]
        if abs(le_y1-le_y2) < 0:
            print("물 줘")
        
        # 입 벌리기
        if abs(mouth_y1 - mouth_y2) > 30:
            print("배고파")

        # 웃기
        elif abs(shape[48,0] - shape[54,0]) > 135:
            print("날씨가 좋아 산책 가자")
    
    cv2.imshow('mouth', mouth_img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break