import threading
import cv2
import time
import random
from matplotlib import image
from sympy import false, true
import threading
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import datetime

def f_dist(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(
        frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255),
                       thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255),
                       thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)

    return frame


def output_keypoints_with_lines(frame, POSE_PAIRS):
    for pairs in POSE_PAIRS :
        cv2.line(frame, points[pairs[0]], points[pairs[1]], (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)), thickness=4, lineType=cv2.LINE_AA)
    return frame


def output_keypoints_with_lines_video(proto_file, weights_file, threshold, BODY_PARTS, POSE_PAIRS):
    
    '''
    stime = time.time()
    while(time.time() - stime < 3.0) :
        print(time.time() - stime)
    '''


    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture('http://192.168.0.5:4747/mjpegfeed')
    capture = cv2.VideoCapture('video/test3.mp4')
    panel = None
    while(True):
        points.clear()
        ret, frame_boy = capture.read()
        frame_boy = cv2.resize(frame_boy,dsize = (0,0), fx = 0.5, fy = 0.5)
        #frame_boy = cv2.resize(frame_boy, (480, 640))
        template = frame_boy.copy()
        #frame_boy = cv2.rotate(frame_boy, cv2.ROTATE_90_COUNTERCLOCKWISE)
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold,
                                     BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        if points[5] and points[2] and points[11] and points[8] and points[4] and points[7] is not None:
            print(points[5])
            print(points[2])
            print(points[11])
            print(points[8])
            print(points[4])
            print(points[7])
            standard = int((points[4][1] + points[7][1]) / 2)
            if(points[5][0] > points[2][0]):
                points[5], points[2] = points[2], points[5]
            if(points[11][0] > points[8][0]):
                points[11], points[8] = points[8], points[11]
            #template = template[min(points[5][1], points[2][1]): max(points[11][1], points[8][1]), points[5][0]: points[2][0]].copy()
            template = template[points[1][1]: max(points[11][1], points[8][1]), points[5][0]: points[2][0]].copy()
            
            break
        print("None")

    #y_gradient = int((max(points[11][1], points[8][1]) - min(points[5][1], points[2][1])) / 2)
    y_gradient = int((max(points[11][1], points[8][1]) - points[1][1])/2)
    x_gradient = int((points[2][0] - points[5][0]) / 2)

    BGR_template = list(range(0, 3))
    BGR_frame_boy = list(range(0, 3))

    res = list(range(0, 3)) 
    BGR_template = cv2.split(template)


    cv2.imshow("test", frame_boy)
    cv2.waitKey()
    cv2.imshow("test2", template)
    cv2.waitKey()

    maxloc = points[14]
    maxloc_t = list(range(0, 3))
    #stime = time.time()
    cnt = 0
    flag = False
    while True:
        ret, frame_boy = capture.read()
        frame_boy = cv2.resize(frame_boy,dsize = (0,0), fx = 0.5, fy = 0.5)
        #frame_boy = cv2.resize(frame_boy, (480, 640))
        #frame_boy = cv2.rotate(frame_boy, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED
        BGR_frame_boy = cv2.split(frame_boy)

        for i in range(0, 3):
            res[i] = cv2.matchTemplate(
                BGR_frame_boy[i], BGR_template[i], cv2.TM_CCOEFF_NORMED)
        _, _, _, maxloc_t[0] = cv2.minMaxLoc(res[0])
        _, _, _, maxloc_t[1] = cv2.minMaxLoc(res[1])
        _, _, _, maxloc_t[2] = cv2.minMaxLoc(res[2])

        tmax = -3.0

        n = maxloc[0]
        m = maxloc[1]
        cv2.rectangle(frame_boy, (n-5, m - 5),
                      (n + 5, m + 5), (255, 255, 255), 2)
        range_gradient = 40
        res_y_max = len(res[0])
        res_x_max = len(res[0][0])
        for i in range(max(n - x_gradient - int(range_gradient/2), 0), min(n - x_gradient + int(range_gradient/2), res_x_max)):
            for j in range(max(0, m - y_gradient - range_gradient, 0), min(m - y_gradient + range_gradient, res_y_max)):
                temp = res[0][j][i] + res[1][j][i] + res[2][j][i]
                if tmax < temp:
                    tmax = temp
                    maxloc = (i + x_gradient, j + y_gradient)
        
        cv2.rectangle(frame_boy, (maxloc[0] - x_gradient, maxloc[1] - y_gradient),
                      (maxloc[0] + x_gradient, maxloc[1] + y_gradient), (255, 255, 255), 2)
        print(str(maxloc[1] - y_gradient) + "             "  + str(standard))
        image = cv2.cvtColor(frame_boy, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if panel is None : 
            panel = tk.Label(root, image = image)
            panel.grid(row=0, column=0)
            panel.image = image
        else :
            panel.configure(image = image)
            panel.image = image

        if flag == False and maxloc[1] - y_gradient  < (standard + 30) - 15 :
            cnt = cnt + 1
            print("cnt : " + str(cnt))
            flag = True
        elif flag == True and maxloc[1] - y_gradient > (standard + 30) + 15 :
            flag = False
        '''
        cv2.rectangle(frame_boy, maxloc_t[0], (maxloc_t[0][0] + x_gradient *
                      2, maxloc_t[0][1] + y_gradient * 2), (255, 0, 0), 2)
        cv2.rectangle(frame_boy, maxloc_t[1], (maxloc_t[1][0] + x_gradient *
                      2, maxloc_t[1][1] + y_gradient * 2), (0, 255, 0), 2)
        cv2.rectangle(frame_boy, maxloc_t[2], (maxloc_t[2][0] + x_gradient *
                      2, maxloc_t[2][1] + y_gradient * 2), (0, 0, 255), 2)
        '''
        update_clock(label_time)
        cv2.imshow("frame_boy", frame_boy)
        if cv2.waitKey(10) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

# 성윤이 코드
def update_clock(label_time):
     time_mm_ss = time.strftime("%H:%M:%S")
     label_time.configure(text=time_mm_ss)
     root.after(1000, update_clock)


if __name__ == "__main__" :
    BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                    15: "Background"}

    POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                    [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

    # 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_mpi = "models/pose/mpi/pose_deploy_linevec.prototxt"

    # 훈련된 모델의 weight 를 저장하는 caffemodel 파일
    weightsFile_mpi = "models/pose/mpi/pose_iter_160000.caffemodel"

    # 키포인트를 저장할 빈 리스트
    points = []
    '''
    output_keypoints_with_lines_video(proto_file=protoFile_mpi, weights_file=weightsFile_mpi,
                                    threshold=0.1, BODY_PARTS=BODY_PARTS_MPI, POSE_PAIRS=POSE_PAIRS_MPI)
    '''
    thread_img = threading.Thread(target = output_keypoints_with_lines_video, args = (protoFile_mpi, weightsFile_mpi, 0.1, BODY_PARTS_MPI, POSE_PAIRS_MPI))
    thread_img.daemon = True
    thread_img.start()

    root = tk.Tk()
    root.title("hello")
    root.geometry("1000x800")

    label_time = tk.Label(root)
    label_time.grid(row=1, column=0)
    time_mm_ss = time.strftime("%H:%M:%S")

    btn_reset = tk.Button(root, text="reset")
    btn_start = tk.Button(root, text="start")
    btn_stop = tk.Button(root, text="stop")

    btn_reset.grid(row=2, column=0)
    btn_start.grid(row=2, column=1)
    btn_stop.grid(row=2, column=2)

    root.mainloop()
