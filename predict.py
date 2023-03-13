import mediapipe as mp
import cv2
import numpy as np
# from scipy import stats
import matplotlib.pyplot as plt
import serial

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path = '/home/pi/Desktop/model_new.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], (1, 30, 44,3))
interpreter.resize_tensor_input(output_details[0]['index'], (1, 2))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

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
    #ser.write(received_data)
    return glove_data

sequence = []
sentence = []
predictions = []
threshold = 0.5
actions =  ['01_hello', '02_nice','03_thank','04_sad','05_want','06_do','07_love','08_me','09_kuy']

mp_hand = mp.solutions.hands#Holistic model
mp_drawing = mp.solutions.drawing_utils #Drawing utilities
import time
cap = cv2.VideoCapture(0)
# Set mediapipe model 
i=0
Time = 0
ser = serial.Serial ("/dev/ttyACM0", 9600)
glove_prev = cv2.getTickCount()
prev = cv2.getTickCount()
glove_data = [0,0,0,0,0,0]
with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as Hands:
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        # Make detections
        image, results = mediapipe_detection(frame, Hands)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, handlm , mp_hand.HAND_CONNECTIONS)
        # ------------------------------1. Glove data ------------------------------
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
        # ------------------------------2. Prediction logic------------------------------
        keypoints = extract_keypoints(image,results , flex , imu)
        sequence.append(keypoints)

        #----------------sliding window to predict-----------------
        if i%15 == 0:
            sequence = sequence[-30:]
            if len(sequence) == 30:
                interpreter.set_tensor(input_details[0]['index'],np.expand_dims(sequence, axis=0).astype('float32'))
                interpreter.invoke()
                #----------------------------predict------------------------------------------
                tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
                prediction_classes = np.argmax(tflite_model_predictions)
                res = tflite_model_predictions[0]

                predictions.append(prediction_classes)
                print(actions[prediction_classes] , len(sentence) )
                data_str_list =str(prediction_classes+1)

                #--------------------------send prediction to stm32---------------------
                ser.write(data_str_list.encode('utf-8'))


                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5: 
                    sentence = [sentence[-1]]
                if len(sentence)>50: sentence = []
        i = i+1

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)

        cv2.imshow('OpenCV Feed', image)
        num_frames = 1
        end = time.time()
        seconds = end - start

        fps  = num_frames / seconds

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
