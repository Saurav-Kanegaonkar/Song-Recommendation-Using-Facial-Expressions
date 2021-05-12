import cv2
import heapq
import webbrowser
import numpy as np
import pandas as pd
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from playsound import playsound


EMOTIONS = ["angry" ,"disgust","fear", "happy", "sad", "surprised", "neutral"]

USE_WEBCAM = True    # If false, loads video file source

# parameters for loading data and images

emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
# hyper-parameters for bounding boxes shape

frame_window = 10       # To be used for the mode comparison with the emotion window

emotion_offsets = (20, 40)   # Hyper parameters for the bounding facial box

# loading models

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')   # face detection model

emotion_classifier = load_model(emotion_model_path)   # 'fer2013' dataset trained model for emotions

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')

video_capture = cv2.VideoCapture(0)       # 0 for default camera or source selection

df = pd.read_csv('Output.csv')


# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()



    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)     # gray image for HOGs, Computationally Easier for Calculation
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)       # color image for video frame

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE) # image scaling

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)    # extracting faces from gray image
        gray_face = gray_image[y1:y2, x1:x2]                                 # storing faces into gray_face
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))       # resizing for emotion detection
        except:
            continue

        gray_face = preprocess_input(gray_face, True)      #converting image into a float 32bit Array
        gray_face = np.expand_dims(gray_face, 0)          #adding 0 axis to the facial float 32-bit array

        gray_face = np.expand_dims(gray_face, -1)             # adding a new axis to the facial 32-bit array
        emotion_prediction = emotion_classifier.predict(gray_face)        # predicting the emotion
        emotion_probability = np.max(emotion_prediction)
        # select the emotion with the maximun probability
        emotion_label_arg = np.argmax(emotion_prediction)              # return the index of the emotion with max probability

        emotion_text = emotion_labels[emotion_label_arg]                # create the emotion label with the label from the index of emotion

        emotion_window.append(emotion_text)                             # add the label to emotion window

        if len(emotion_window) > frame_window:                         # limiting the list to 10 items only
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)                # find the mode of the emotion window items
        except:
            continue

        if emotion_text == 'angry':                                      # giving the text colors with numpy (R,G,B) values
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)           # finally drawing the the bounding box

        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)                                  # adding the emotion text


        canvas = np.zeros((250, 350, 3), dtype="uint8")
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, emotion_prediction[0])):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        cv2.imshow("Probabilities", canvas)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)


    if cv2.waitKey(1) & 0xFF == ord('p'):                            # exit conditions
        pred=emotion_prediction[0]
        emo = heapq.nlargest(3, range(len(pred)), key=pred.__getitem__)
        emotion_text = emotion_labels[emo[0]]
        emotion_text1 = emotion_labels[emo[1]]
        emotion_text2 = emotion_labels[emo[2]]
        emotions=[emotion_text,emotion_text1,emotion_text2]
        playlist=10
        emo_vals=heapq.nlargest(3, pred)
        for i in range(len(emo_vals)):
            emo_vals[i]=round(emo_vals[i]*10)
        emo_vals[2]=10-emo_vals[0]-emo_vals[1]
        print(emo_vals)
        rows=[]
        for i in range(len(emotions)):
            count=0
            for j in range(df.shape[0]):
                if df['Face Emotion'][j] == emotions[i]:
                    if count< emo_vals[i]:
                        rows.append(j)
                        count=count+1
                    else:
                        break
        print(emotions)
        print(df.loc[rows, :])

        webbrowser.open(df.loc[rows[0],'Youtube Link'])
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):                            # exit conditions
        print(emotion_text)
        playsound('cool.mp3')
        break

cap.release()
cv2.destroyAllWindows()