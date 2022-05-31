from pipes import Template
import cv2
import numpy as np
import time


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
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)

    return frame


def output_keypoints_with_lines_video(proto_file, weights_file, threshold, BODY_PARTS, POSE_PAIRS):

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    capture = cv2.VideoCapture(0)

    while(True):
        points.clear()
        ret, frame_boy = capture.read()
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        gray_frame_boy = cv2.cvtColor(frame_boy, cv2.COLOR_RGB2GRAY)

        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold,
                                     BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(
            frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        if points[14] is not None:
            break
        print("None")

    # ---------------------------
    temp_gradient = 50
    tuple_gradient = (temp_gradient, temp_gradient)
    # ---------------------------
    print(points[14])

    gray_template = gray_frame_boy[points[14][1] - temp_gradient: points[14][1] + temp_gradient,
                                   points[14][0] - temp_gradient: points[14][0] + temp_gradient].copy()

    template = frame_boy[points[14][1] - temp_gradient: points[14][1] + temp_gradient,
                         points[14][0] - temp_gradient: points[14][0] + temp_gradient].copy()

    BGR_template = list(range(0, 3))
    BGR_frame_boy = list(range(0, 3))
    res = list(range(0, 3))
    BGR_template = cv2.split(template)

    cv2.imshow("test", frame_boy)
    cv2.waitKey()
    cv2.imshow("test2", gray_template)
    cv2.waitKey()

    maxloc = points[14]
    maxloc_t = list(range(0, 3))
    while True:
        ret, frame_boy = capture.read()
        #cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED
        BGR_frame_boy = cv2.split(frame_boy)

        for i in range(0, 3) :
            res[i] = cv2.matchTemplate(BGR_frame_boy[i], BGR_template[i], cv2.TM_CCOEFF_NORMED)
        
        _, _, _, maxloc_t[0] = cv2.minMaxLoc(res[0])
        _, _, _, maxloc_t[1] = cv2.minMaxLoc(res[1])
        _, _, _, maxloc_t[2] = cv2.minMaxLoc(res[2])

        tmax = -3.0
        
        n = maxloc[0]
        m = maxloc[1]
        cv2.rectangle(frame_boy, (n, m), (n + 10, m + 10), (255, 255, 255), 2)
        for i in range(m - 10, m + 10) :
            for j in range(n - 10, n + 10) :
                temp = 0.0
                for k in range(0, 3) :
                    temp += res[k][i][j]
                if tmax < temp :
                    maxloc = (i, j )  
                    tmax = temp
        

        #cv2.rectangle(frame_boy, (n, m), (n, m), (0, 0, 0) , 2)

        cv2.rectangle(frame_boy, maxloc, (maxloc[0] + temp_gradient * 2, maxloc[1] + temp_gradient * 2), (255, 255, 255), 2)


        cv2.rectangle(frame_boy, maxloc_t[0], (maxloc_t[0][0] + temp_gradient * 2, maxloc_t[0][1] + temp_gradient * 2), (255, 0, 0), 2)
        cv2.rectangle(frame_boy, maxloc_t[1], (maxloc_t[1][0] + temp_gradient * 2, maxloc_t[1][1] + temp_gradient * 2), (0, 255, 0), 2)
        cv2.rectangle(frame_boy, maxloc_t[2], (maxloc_t[2][0] + temp_gradient * 2, maxloc_t[2][1] + temp_gradient * 2), (0, 0, 255), 2)


        cv2.imshow("frame_boy", frame_boy)
        if cv2.waitKey(10) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                  [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [
                          15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_mpi = "./Github/HCI/models/pose/mpi/pose_deploy_linevec.prototxt"
protoFile_mpi_faster = "./Github/HCI/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
protoFile_coco = "./Github/HCI/models/pose/coco/pose_deploy_linevec.prototxt"
protoFile_body_25 = "./Github/HCI/models/pose/body_25/pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_mpi = "./Github/HCI/models/pose/mpi/pose_iter_160000.caffemodel"
weightsFile_coco = "./Github/HCI/models/pose/coco/pose_iter_440000.caffemodel"
weightsFile_body_25 = "./Github/HCI/models/pose/body_25/pose_iter_584000.caffemodel"

# 키포인트를 저장할 빈 리스트
points = []

output_keypoints_with_lines_video(proto_file=protoFile_mpi, weights_file=weightsFile_mpi,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_MPI, POSE_PAIRS=POSE_PAIRS_MPI)

'''
output_keypoints_with_lines_video(proto_file=protoFile_coco, weights_file=weightsFile_coco, 
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_COCO, POSE_PAIRS=POSE_PAIRS_COCO)
'''
'''
output_keypoints_with_lines_video(proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)
'''
