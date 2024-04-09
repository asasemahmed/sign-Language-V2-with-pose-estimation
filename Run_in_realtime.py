"""
run the model in real time
"""

import mediapipe as mp
import cv2
import numpy as np
import pickle

file2 = open('random_ft.pkl', 'rb')
model = pickle.load(file2)


# the number of device
cap = cv2.VideoCapture(2)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    rep, frame = cap.read()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    zeros = list(np.zeros((21), dtype=int))
    if results.multi_hand_landmarks:
        cxcy = []
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cxcy.append(cx+cy)
                # print([id, cx, cy], '\n')
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        if len(cxcy) < 40:
            text = model.predict(np.array(cxcy+zeros).reshape(1, 42))[0]
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (50, 0, 255), 2)
        else :
            text = model.predict(np.array(cxcy).reshape(1, 42))[0]
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (50, 0, 255), 2)



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()