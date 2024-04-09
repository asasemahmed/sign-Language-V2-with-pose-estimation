"""
Collecting Data for the model
"""

import mediapipe as mp
import cv2
import csv
import numpy as np


file = open('xdata.csv', 'a', newline='')

writer = csv.writer(file)

# # rum this code one time to create the header columns
# string = ''
# for n in range(42):
#     if n > 20:
#         n -= 21
#         string += f"x2_{n} "
#         continue
#     string += f"x1_{n} "
# writer.writerow(string.split()+['label'])


cap = cv2.VideoCapture(2)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

label = 'I Love You'
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
            writer.writerow(cxcy+zeros + [label])
        else :
            writer.writerow(cxcy + [label])



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
   
    