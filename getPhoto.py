import numpy as np
import cv2
import os
import shutil
from playsound import playsound

# bufferless VideoCapture
import queue, threading
# camUrl = 'rtsp://admin:test2019@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0'
camUrl = 0

class VideoCapture:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
        ret, frame = self.cap.read()
        if (ret): 
            if (not self.q.empty()):
                try: self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty: pass
            self.q.put(frame)
        else: print('Skip frame')
  def read(self):
    return self.q.get()

def createUserDataset(name,trExist):
    net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
    cap = VideoCapture(camUrl)
    sampleN=0
    cnt = 20
    if(trExist):
        try:
            photosNumbers = []
            for fotoName in os.listdir("static/PhotoSeries/"+name): photosNumbers.append(int(fotoName.split(".")[1]))
            sampleN = max(photosNumbers)
        except: 
            print("Error file name for user: "+name)
            sampleN = 100
    else: os.mkdir("static/PhotoSeries/"+name)
    while 1:
        frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (103.93, 116.77, 123.68)) #  (104.0, 177.0, 123.0)) #
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[0]):
            if (detections[0, 0, i, 2] > 0.6): 
                fh, fw = frame.shape[:2]
                (x1,y1,x2,y2) = (detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int")
                sampleN=sampleN+1
                face = frame[y1:y2, x1:x2]
                cv2.imwrite("static/PhotoSeries/"+name+'/'+name+ "." +str(sampleN)+ ".jpg", face)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cnt -=1
        playsound('data/photoOk.wav')
        cv2.imshow('img',frame)
        cv2.waitKey(300)
        cv2.waitKey(1)
        if (cnt<1): break
    del(cap)

name = input("Please input user name?")
name = str(name)
trExist = os.path.isdir('static/PhotoSeries/'+name)
if(trExist):
    tr = input("User '{}' already exists. Do you want add new photos? y(Yes) n(No)".format(name))
    if(tr.lower()=='y'): createUserDataset(name,trExist)
    elif(tr.lower()=='n'): 
        tr = input("User '{}' already exists. Do you want replace with new photos? y(Yes) n(No)".format(name))
        if(tr.lower()=='y'): 
            try: shutil.rmtree('static/PhotoSeries/'+name)
            except: print("Error! Can not delete user photos.")
            createUserDataset(name,False)
        else: print("exit")
    else: print("exit")
else: 
    print("creating new user dataset")
    createUserDataset(name,trExist)

