import mediapipe as mp
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('one_two.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# print(x.head())
la = LabelEncoder()
y = la.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)


rn = RandomForestClassifier()
rn.fit(x_train, y_train)
print(f"score -> {rn.score(x_test, y_test)}")

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



while True:
    rep, frame = cap.read()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    cxcy = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cxcy.append(cx+cy)

        prd = rn.predict(np.array([cxcy[:21]]))
        if prd[0] == 0:
            frame = cv2.putText(frame,
                    'one',
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)
        elif prd[0] == 1:
            frame = cv2.putText(frame,
                    'two',
                    (80, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)

        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()