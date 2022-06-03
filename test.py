import cv2

cap = cv2.VideoCapture('http://192.168.220.63:4747/mjpegfeed')

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

while(True):
    ret, frame = cap.read()    # Read 결과와 frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    if(ret) :
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        cv2.imshow('frame_gray', gray)    # Gray 화면 출력
        if cv2.waitKey(1) == ord('q'):  
            break
cap.release()
cv2.destroyAllWindows()