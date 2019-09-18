#Face Recognition with Google's Facenet Model
#Author Sefik Ilkin Serengil (sefiks.com)
#You can find the documentation of this code from the following link: 
#https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/
#Tested for TensorFlow 1.9.0, Keras 2.2.0 and Python 3.5.5
#-----------------------
import numpy as np
import cv2
import time
import random
from keras.models import Model, Sequential
# from keras.models import model_from_json
from keras.models import load_model
# from theano import tensor as T

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import keras.backend as K
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir
# bufferless VideoCapture
import queue, threading

#-----------------------
# camUrl = 'rtsp://admin:test2019@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0'
camUrl = 0
folderPathTrain = "static/PhotoSeries/"
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
        # else: print('Skip frame')
  def read(self):
    return self.q.get()

def addPhotoSeriesQueue(emb,name,img):
	print('save new unknown face '+name)
	path = 'static/PhotoSeriesQueue'

cap = VideoCapture(camUrl)
#-----------------------
required_size=(160, 160)
font = cv2.FONT_HERSHEY_DUPLEX
# face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
modelFaceNet = load_model('data/facenet_keras.h5')
data = np.load('data/faces-embeddings.npz')

threshold = 21 #tuned threshold for l2 disabled euclidean distance
employees = dict()
print("start")
embds, names = data['arr_0'], data['arr_1']
# for face, label in zip(embds, lbs): employees[label] = face
print(len(embds), len(names))
unFace_embs = []
unFace_cnts = []
unFace_saved_embs = []
unFace_saved_names = []
newDetectedIdSlow = {}
while(True):
	timers_start = time.time()
	frame = cap.read()
	# faces = face_cascade.detectMultiScale(img, 1.3, 5)
	blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (103.93, 116.77, 123.68))
	net.setInput(blob)
	fh, fw = frame.shape[:2]
	detections = net.forward()
	txt =''
	# for (x,y,w,h) in faces:
	# 	if w > 130: #discard small detected faces
	for i in range(detections.shape[0]):
		if (detections[0, 0, i, 2] > 0.5):
			x1,y1,x2,y2 = (detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int")
			face = frame[y1:y2, x1:x2]
			face = cv2.resize(face,required_size)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			# print(type(face[0][0][0]))
			face = face.astype('float32')
			cv2.rectangle(frame, (x1,y1), (x2,y2), (67, 67, 67), 1) #draw rectangle to main image
			faceNorm = (face - face.mean()) / face.std()
			samples = np.expand_dims(faceNorm, axis=0)
			# face_name = model.predict(samples)
			new_emb = np.expand_dims(modelFaceNet.predict(samples)[0], axis=0)
			# captured2 = K.expand_dims(face_class, axis=0)
			distancesUn = []
			# check if exist in saved before uknown
			for emb in embds: distancesUn.append(np.sqrt(np.sum(np.square(emb - new_emb))))
			index1 = np.argmin(distancesUn)
			txt += 'appear {}({})'.format(names[index1],distancesUn[index1])
			if (distancesUn[index1] > 10): txt+=">10!!!!"
			print (txt)
	fps = str(round(1.0/(time.time()-timers_start),1)) 
	# nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	txt_fps = 'fps={} {}'.format(fps,txt)
	print(txt_fps)
	cv2.putText(frame,txt_fps,(20, 20), font, 0.4,(5,5,255),1,cv2.LINE_AA)
	cv2.imshow('img',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
#kill open cv things		
cap.release()
cv2.destroyAllWindows()