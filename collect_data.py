import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import serial
# cap = cv2.VideoCapture("udpsrc port=5000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! ximagesink")#device number
cap = cv2.VideoCapture(0)
break_out_flag =False
mp_hand = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils #Drawing utilities

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def extract_keypoints(img , results , flex , imu):
    lh = 0
    rh = 0
    if results.multi_hand_landmarks :
        hand_joint12 = [] #-----list of All joints
        for handlm in results.multi_hand_landmarks:
            hand_joint = []
            for id, lm in enumerate(handlm.landmark):
                hand_joint.append([lm.x,lm.y,lm.z]) #-----list of 1 joint

            hand_joint12.append(hand_joint)
        if len(results.multi_hand_landmarks)==2:
            lh = np.array(hand_joint12[0])
            rh = np.array(hand_joint12[1])
        elif len(results.multi_hand_landmarks)==1: 
            lh = np.array(hand_joint12[0])
            rh = np.zeros((21,3))
    else: 
        lh = np.zeros((21,3))
        rh = np.zeros((21,3))
    hand_joint = np.vstack((lh,rh, flex , imu))
    return hand_joint

    

def extract_glove(glove_data):
    received_data = ser.read()
    data_left = ser.inWaiting()             #check for remaining byte
    received_data += ser.read(data_left)
    received_data2 = received_data.decode('utf-8')
    received_data2 = received_data2.split(",")
    try:
        if len(received_data2)==7:  #clean '555'
            glove_data = []
            for i in received_data2[1:]: #clean '555'
                res = float(i)
                glove_data.append(res)
    except:
        pass
    ser.write(received_data)
    return glove_data
pose = '08_me' # change if change new pose
data_path = os.path.join('data')

actions = np.array([pose])
# collect 30 video
start_video = 60  # change  if get more data
stop_video = 100
#each video 30 frame
sequence_lenght = 30

#-------------------define for gloves--------------------
Time = 0
ser = serial.Serial ("/dev/ttyACM0", 9600)
glove_prev = cv2.getTickCount()
prev = cv2.getTickCount()
glove_data = [0,0,0,0,0,0]
for action in actions:
    for sequence in range(start_video,stop_video):
        try:
            os.makedirs(os.path.join(data_path,action,str(sequence)))
        except:
            pass
with mp_hand.Hands( max_num_hands=2,min_detection_confidence=0.25, min_tracking_confidence=0.25) as Hands:
    for action in actions:
        for sequence in range(start_video,stop_video):
            for frame_num in range(sequence_lenght):
                start = time.time()
                ret,frame = cap.read()

                #------------------------detect--------------------------------------------
                image,results = mediapipe_detection(frame,Hands)
                #------------------------draw--------------------------------------------
                if results.multi_hand_landmarks:
                    for handlm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, handlm , mp_hand.HAND_CONNECTIONS)
                if frame_num ==0:
                    cv2.putText(image,'start get data',(120,200),
                               cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    # cv2.putText(image,'action {} Video {}'.format(action,sequence),(15,12),
                    #            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed',image)
                    cv2.waitKey(2000)
                else:
                    # cv2.putText(image,'action {} Video {}'.format(action,sequence),(15,12),
                    #            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed',image)
                cv2.waitKey(1)
                # cv2.imshow("sdsdsd",image)
                #------------------------get datagloves--------------------------------------------
                glove_new = cv2.getTickCount()
                if Time>1.5:
                    glove_data_check = extract_glove(glove_data)
                    if glove_data_check != None:
                        glove_data = glove_data_check
                    glove_prev = glove_new
                if len(glove_data)!=6: glove_data = [0,0,0,0,0,0]
                flex = np.array(glove_data[:3])
                imu = np.array(glove_data[3:6])
                Time = (glove_new-glove_prev)/cv2.getTickFrequency()
                # print(glove_data)

                #------------------------cal fps--------------------------------------------
                num_frames = 1
                end = time.time()
                seconds = end - start
                # print ("Time taken : {0} seconds".format(seconds))
                fps  = num_frames / seconds
                # print("Estimated frames per second : {0}".format(fps))

                #------------------------extract join x y z--------------------------------------------
                keypoints = extract_keypoints(image , results , flex , imu)
                npy_path = os.path.join(data_path,action,str(sequence))
                npy_path = os.path.join(npy_path , str(frame_num))
                np.save(npy_path,keypoints)
                ##------------------------Break gracefully--------------------------------------------Break gracefully
                # if cv2.waitKey(10) & 0xFF==ord('q'):
                if cv2.waitKey(10) & 0xFF==27:
                    break_out_flag =True
                    break
            if break_out_flag:
                break
        if break_out_flag:
             break
    cap.release()
    cv2.destroyAllWindows()
