import cv2

cap = cv2.VideoCapture(0)
cnt = 0

while(True):
    ret, frame = cap.read()    # Read 결과와 frame
    cnt = cnt + 1
    print(cnt)
    if(ret) :
        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
