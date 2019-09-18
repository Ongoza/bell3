from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# from sklearn.metrics.pairwise import euclidean_distances
from keras.models import load_model
# from matplotlib import pyplot
import cv2
import numpy as np
import time

# prepare a camera
# camUrl = 'rtsp://admin:test2019@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0'
camUrl = 0

# bufferless VideoCapture
import queue, threading

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

cap = VideoCapture(camUrl)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
font = cv2.FONT_HERSHEY_DUPLEX
# prepare face detection
net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
print("face detection is ready")
# prepare face recognation
modelFaceNet = load_model('data/facenet_keras.h5')
data = np.load('data/faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
required_size=(160, 160)
skip = 1
counter = skip
print("face detection is ready")
while True:
    frame = cap.read()
    if(counter!=0): counter-=1
    else:
        counter = skip
        timers_start = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (103.93, 116.77, 123.68))
        net.setInput(blob)
        fh, fw = frame.shape[:2]
        detections = net.forward()
        txt = ''
        for i in range(detections.shape[0]):
            if (detections[0, 0, i, 2] > 0.5): 
                x1,y1,x2,y2 = (detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int")
                face = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2],required_size), cv2.COLOR_BGR2RGB)
                face = face.astype('float32')
                face = (face - face.mean()) / face.std()
                new_embd = np.expand_dims(modelFaceNet.predict(np.expand_dims(face, axis=0))[0], axis=0)
                face_class = model.predict(new_embd)
                face_prob = model.predict_proba(new_embd)
                # print(face_class,face_prob)
                conf = round(face_prob[0,face_class[0]] * 100)
                idFace = out_encoder.inverse_transform(face_class)[0]
                txt = "{}({:n}%)".format(idFace, conf)
                if(conf<85): txt = "Unknown. Nearest is:"+txt
                print(txt)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame,txt,(x1+2, y1+15), font, 0.4,(5,255,5),1,cv2.LINE_AA)
        fps = str(round(1.0/(time.time()-timers_start),1)) 
        # nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        txt_fps = 'fps={} {}'.format(fps,txt)
        print(txt_fps)
        cv2.putText(frame,txt_fps,(20, 20), font, 0.4,(5,5,255),1,cv2.LINE_AA)
        cv2.imshow("testCamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

if(cap):
    if(cap.isOpened): cap.release()
cv2.destroyAllWindows()

# You can use another layer of authentication to the output class. 
# Suppose your classifier return 'xyz' as output class. 
# You can then double check by comparing the input image and image which is in your data set. 
# Best way will be to compare the embedding of those two image and check for threshold distance.

# KNN queries can be slow with large datasets. 
# There are approximate nearest neighbor algorithms which work reasonably well and much faster than KNN; 
# see LSH Forests if you are using sklearn.