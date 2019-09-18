import os
import numpy as np
import cv2
from playsound import playsound

def createUserDataset(name):
    # face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('data/intelFace.xml')
    cap = cv2.VideoCapture(0)
    sampleN=0
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1,15)
        for (x,y,w,h) in faces:
            sampleN=sampleN+1
            face = gray[y:y+h, x:x+w]
            print(y+h,x+w)
            # face = cv2.resize(face,(340,340), interpolation = cv2.INTER_AREA)
            cv2.imwrite("static/PhotoSeries/"+str(name)+'/'+str(name)+ "." +str(sampleN)+ ".jpg", face)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            playsound('data/photoOk.wav')
            cv2.imshow('img',img)
        cv2.waitKey(200)
        cv2.waitKey(1)
        if sampleN > 20:
            break
    cap.release()
    cv2.destroyAllWindows()

name = input("Please input user name? ")
if(os.path.isdir('static/'+name)):
    tr = input("User '{}' already exists. Do you want add new photos? y(Yes) n(No)".format(name))
    if(tr=='y'): createUserDataset(name)
    elif(tr=='n'): print("exit")
    else: print("exit")
else: 
    print("creating new user datset")
    createUserDataset(name)

