import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read() #이미지 읽어오기

    if not ret:
        break
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break