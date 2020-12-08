import cv2
import numpy as np
import winsound

from tensorflow import keras

haar_cascade=cv2.CascadeClassifier('haar_face.xml')
model=keras.models.load_model("Mask_detector.model")
print("Everything in")

freq=2500
duration=35
capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if flag:
        faces=haar_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3)
        for x,y,w,h in faces:
            face = img[y:y + h, x:x + w]
            # print(face.shape)
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(face, axis=0)
            #print(face.shape)
            # model(np.reshape(0, [1, 1]))
            face = face / 255.0
            pred = model.predict(face)
            # print("Pred: ", pred)
            pred = int(pred)
            if pred == 1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
                cv2.putText(img, "No Mask", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                winsound.Beep(freq,duration)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(img, "Mask On", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('result',img)
        # 27 - ASCII of Escape
        if cv2.waitKey(2)==27:
            break

capture.release()
cv2.destroyAllWindows()
