# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC
# from keras.models import load_model

import threading
import cv2
import numpy as np
import time
import random
# import face_recognition
import os
import logging
import sys, traceback
import requests
import json
# создать папку для хранения неизвестных лиц
# писать серию фото неизвестных из Н кадров
# указать после попадания на сколько кадров неизвестный отправляется на сервер
# создать папку объектов, которые на фиксируются (типовые ошибки для камеры)
# хранить Н полных кадров - сколько указать в настройках
# сделать циклы на нампи
# файл хранения конфигурации камеры
# логи
# bufferless VideoCapture
import queue, threading

import tensorflow as tf
from tensorflow.python.platform import gfile
import detect_face
import pickle

import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# camUrl = 'rtsp://admin:test2019@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0'

class VideoCapture:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    self._stopevent = threading.Event()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while not self._stopevent.isSet():
        if(self.cap):
            ret, frame = self.cap.read()
            if (ret): 
                if (not self.q.empty()):
                    try: self.q.get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty: pass
                self.q.put(frame)
  def read(self):
    return self.q.get()
  def exit(self):
      self._stopevent.set()
      if(self.cap):
          if(self.cap.isOpened()):self.cap.release()

class Classify():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.sess = tf.Session().__enter__()
        self.load_model("data/20180402-114759/20180402-114759.pb")
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        self.emb_array = np.zeros((1, self.embedding_size))
        # data = np.load('data/faces-embeddings.npz')
        # self.model, self.class_names = data['arr_0'], data['arr_1']
        classifier_filename_exp = os.path.expanduser("data/myClassifier.pkl")
        with open(classifier_filename_exp, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile)

    def __del__(self):
        self.sess.close()

    def predict(self, face):
        image = self.load_image(face)
        self.feed_dict = { self.images_placeholder:image, self.phase_train_placeholder:False }
        self.emb_array[0,:] = self.sess.run(self.embeddings, feed_dict=self.feed_dict)
        predictions = self.model.predict_proba(self.emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        return self.class_names[best_class_indices[0]], int(best_class_probabilities[0]*100), self.emb_array

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  
    
    def load_image(self, face):
        image = np.zeros((1, 160, 160, 3))
        # img = imageio.imread(image_path)
        img = self.prewhiten(face)
        image[0,:,:,:] = img
        return image

    def load_model(self, model="data/20180402-114759"):
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=None, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self.get_model_filenames(model_exp)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            self.saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=None)
            self.saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

class Camera(threading.Thread):
    def __init__(self, id, url, serverUrl, config=None):
        if(id):
            self.isStarting = True
            self.isRun = False
            self.id = id
            self.log_divider = '##'
            self.fromaddr = 'oleg2231710@gmail.com'
            self.toaddr = ['oleg@ongoza.com']
            self.password = 'z'
            self.log = logging.getLogger('Camera_' + id)
            formatter = logging.Formatter('%(asctime)s'+self.log_divider+'%(levelname)s'+self.log_divider+'%(message)s')
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)
            fileHandler = logging.FileHandler('static/logs/camera_' + id + '.log', mode='a+')
            fileHandler.setFormatter(formatter)
            self.log.setLevel("DEBUG")
            self.log.addHandler(fileHandler)
            self.serverUrl = serverUrl
            self.log.addHandler(streamHandler)
            self.config = config
            # self.delayBeforeLeave = 4
            self.classifiers = ['data/haarcascade_frontalface_default.xml',
                'data/haarcascade_frontalface_alt.xml',
                'data/haarcascade_frontalface_alt2.xml',
                'data/haarcascade_frontalface_alt_tree.xml',
                'data/intelFace.xml'
                ]
            self.startMsg = "restarted"
            if(config==None):
                self.startMsg = "started"
                cfgFile = 'configs/'+id+'.json'
                if(os.path.isfile(cfgFile)):
                    with open(cfgFile) as jf:
                        self.config = json.load(jf)
                else:
                    self.config = {
                        "cId": id, # camera id
                        "cName": "Cam", # camera name
                        "cUrl": url, # camra adddress 0 for USB
                        "vdScl": 1, # scale video
                        "cConn": 3, # try connect to camera - not used!!!!!!!!!!
                        "frSkip": 1, # frames for sciping. more for less GPU or CPU usage
                        "isLog":True, # does save log to file?
                        "isDetect":True, # does detect face in video?
                        "imgBrdr": 10, # resize border around fac - not used
                        "frScl": 1.1, #  
                        "minNbrs": 5, # parameter for opencv face detection: min distance to neighbors 
                        "minSz": 30, # parameter for opencv face detection: min face size
                        "dtctInd": 5, # index of detection algoritms - 5 for DNN 
                        "imgFrmt": ".webp", # file format for store
                        "imgQlt": 95, # file format image quality  for store
                        "fcCmpr": True, # does recognize faces?
                        "fcCmprVal": 80, #  min face recognation probality level
                        "fcUnCmprVal": 20, # max face recognation closes level between faces 
                        "fcFrDel":2, # wait n frames before disapear alert
                        "fcFrAdd":2# wait n frames before appear alert
                    }
            self.tryCounter = self.config['cConn']
            self.cap = None
            self.img = None
            self.timers= {"startCamera":time.time()}
            self.unFaceSavedEmbs = []
            self.unFaceSavedNames = []
            self.knFaceEmbs = []
            self.knFaceNames = []
            self.names_for_appear = {}
            self.font = cv2.FONT_HERSHEY_DUPLEX
            self._stopevent = threading.Event()
            self.isRestart = False
            self.stopDelay = False
            # self.slowNames = {}
            self.reloadConfigItems = ['camUrl','isCompare','classifierIndex']
            self.log.info("start camera config ok")
            threading.Thread.__init__(self)
        else: self.log.error("Camera name can not be empty.")
    
    # def loadFaces(self):
    #     location_faces = os.path.join(os.getcwd(), "static/faces/")
    #     self.log.info("Start loading photos from "+location_faces)
    #     count = 0
    #     count_err = 0
    #     for fileName in os.listdir(location_faces):
    #         abs_file_path = os.path.join(location_faces, fileName)
    #         try:
    #             picture = face_recognition.load_image_file(abs_file_path, mode='RGB')
    #             encoding = face_recognition.face_encodings(picture)[0]
    #             self.knownFaceEncodings.append(encoding)
    #             self.knownFaceNames.append([os.path.splitext(fileName)[0],fileName])
    #             count += 1
    #         except:
    #             count_err += 1
    #             self.log.error("error open file: <img src=\"faces/" + str(fileName) + "\"/>")
    #     self.log.info("Knowned faces loaded successefull. Faces: " + str(count) + ". Errors: " + str(count_err))
    #     self.config['facesInMonitoring'] = count
    #     self.config['facesSkipMonitoring'] = count_err
    
    def run(self):
        self.log.info("start camera")
        self.isStarting = True
        self.cap = VideoCapture(self.config['cUrl'])
        # if(self.config['isCompare']): self.loadFaces()
        # if(self.config['isCompareDNN']):
        #     for imagePath in sorted(os.listdir("static/PhotoSeries/")):
        #         id = os.path.split(imagePath)[-1].split(".")[0]
        #         name = os.path.split(imagePath)[-1].split(".")[1]
        #         if not id in self.slowNames.keys(): 
        #             self.slowNames[id] = [name,imagePath]
        #             # print(id,imagePath)
            # print(self.slowNames)
        self.get_frame()
    
    def exit(self):
        # self.log.info("pause camera")
        try:
            self._stopevent.set()
            self.stopDelay = False
            self.startMsg = "started"
            self.img = None
            self.isStarting = False
            self.tryCounter = self.config['cConn']            
            self.isRun = False
            if(self.cap): self.cap.exit()
            time.sleep(2)
            self.cap = None
            print("closed camera object")
        except Exception as e: 
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,  limit=2, file=sys.stdout)
            print(e)
    
    def addPhotoSeriesQueue(self,emb,name,img):
        print('save new unknown face '+name)
        path = 'static/PhotoSeriesQueue'
        self.unFaceSavedEmbs.append(emb)
        self.unFaceSavedNames.append(name)
    
    def get_frame(self):
        if(self.cap):
            cnt = self.config['frSkip']
            # imgWebCompression = []
            # if(self.config['imgWebFormat']=='.png'): imgWebCompression = [cv2.IMWRITE_PNG_COMPRESSION, self.config['imgWebQuality']/10]
            # elif(self.config['imgWebFormat']=='.jpg'): imgWebCompression = [cv2.IMWRITE_JPEG_QUALITY, self.config['imgWebQuality']]
            # else: imgWebCompression = [cv2.IMWRITE_WEBP_QUALITY, self.config['imgWebQuality']]
            counter = 0
            net = None
            # newDetectedId = {}
            curDetectedIds = {}
            trFaseCascade = False
            required_size=(160, 160)
            unFace_embs = []
            unFace_cnts = []
            # names_for_appear = {}
            # unknown_face_names = []
            # yScale = int(self.config['faceBorder']*3)
            print("classifier index="+str(self.config['dtctInd']))
            if(self.config['dtctInd']==5):  net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
            else:
                faceCascade = cv2.CascadeClassifier(self.classifiers[self.config['dtctInd']])
                trFaseCascade = True
            # prepare face recognation
            if(self.config['fcCmpr']):
                classifier = Classify()
                # modelEmb = load_model('data/facenet_keras.h5')
                # data = np.load('data/faces-embeddings.npz')
                # embds, names = data['arr_0'], data['arr_1']
                # in_encoder = Normalizer(norm='l2')
                # self.knFaceEmbs = in_encoder.transform(embds)
                # out_encoder = LabelEncoder()
                # out_encoder.fit(names)
                # self.knFaceNames = out_encoder.transform(names)
                # model = SVC(kernel='linear', probability=True)
                # model.fit(self.knFaceEmbs, self.knFaceNames)
            if(not self.stopDelay):
                while not self._stopevent.isSet():
                    counter += 1 if(counter<10000) else 0
                    frame = self.cap.read()
                    try:
                        if (cnt > 0): cnt -= 1
                        else:
                            temp_timers = {"N" : counter,"start" : time.time()}
                            cnt = self.config['frSkip']
                            trNewEvent = False
                            events_data = []
                            if(self.config['frScl']!=1): frame = cv2.resize(frame, (0, 0), fx=self.config['frScl'], fy=self.config['frScl'])
                            if(self.config['isDetect']):
                                fh, fw = frame.shape[:2]
                                faceLocations = []
                                if(trFaseCascade): 
                                    #faceLocations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn") 
                                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    faceLocations = faceCascade.detectMultiScale(frame_gray,scaleFactor=self.config['scaleFactor'],minNeighbors=self.config['minNeighbors'],minSize=(self.config['minSize'],self.config['minSize']),flags=cv2.CASCADE_SCALE_IMAGE) #
                                else: 
                                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (103.93, 116.77, 123.68))
                                    # blob = cv2.dnn.blobFromImage(frame, 0.00392, (300,300), (0,0,0), True, crop=False)
                                    net.setInput(blob)
                                    detections = net.forward()
                                    for i in range(detections.shape[0]):
                                        if (detections[0, 0, i, 2] > self.config['dtThr']): faceLocations.append((detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int"))
                                temp_timers["detect"] = round(time.time()-temp_timers["start"],4)
                                #print(faceLocations)
                                for (x1, y1, x2, y2) in faceLocations:
                                    # create bigger arrea for a face photo
                                    # border = self.config['faceBorder']
                                    # bx1=0 if(x1<border) else x1-border
                                    # by1=0 if(y1<border) else y1-border
                                    # bx2=fw if((x2+border)>fw) else x2+border
                                    # by2=fh if((y2+border)>fh) else y2+border
                                    # temp_facePhotos.append(frame[y0:yh, x0:xw])
                                    # temp_faceSources.append(frame_gray[y:y+h, x:x+w])
                                    # locName = bx1+2, by1+15
                                    locName = x1+2, y1+15
                                    # faceLocationsNames.append(locName)
                                    # faceLocationsNames.append((x1,x2))
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    # cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                    temp_timers["location"] = round(time.time()-temp_timers["start"],4)
                                    if(self.config['fcCmpr']):
                                        # print(x1, y1, x2, y2)
                                        faceNameText = ''
                                        face = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2],required_size), cv2.COLOR_BGR2RGB)
                                        # face = face.astype('float32')
                                        # faceNorm = (face - face.mean()) / face.std()
                                        # new_emb = np.expand_dims(modelEmb.predict(np.expand_dims(faceNorm, axis=0))[0], axis=0) # get embeding for current face
                                        temp_timers["embeding"] = round(time.time()-temp_timers["start"],4)
                                        idFace, conf , new_emb = classifier.predict(face)
                                        # print(idFace, conf)
                                        # face_class = model.predict(new_emb)  # try find face in known faces list
                                        # face_prob = model.predict_proba(new_emb) # get recognation probability 
                                        # conf = int(face_prob[0,face_class[0]]*100) 
                                        # idFace = out_encoder.inverse_transform(face_class)[0]
                                        print(idFace,conf)
                                        if(conf>self.config['fcCmprVal']): #if probality of face recognation is enough big
                                            if(not idFace in curDetectedIds.keys()): curDetectedIds[idFace] = [self.config['fcFrAdd'],self.config['fcFrDel']]
                                            if(curDetectedIds[idFace][0] == 0):
                                                trNewEvent = True 
                                                events_data.append(['0',idFace,'({}%)'.format(conf),'PhotoSeries',str(counter)])
                                            curDetectedIds[idFace] = [curDetectedIds[idFace][0]-1,self.config['fcFrDel']]
                                            temp_timers["name1"] = round(time.time()-temp_timers["start"],4)
                                        else: # try find face in unknown saved faces list
                                            faceNameText += 'closest known face is {} ({}%)'.format(idFace,conf)
                                            if(self.unFaceSavedEmbs):
                                                distancesUn = []
                                                for emb in self.unFaceSavedEmbs: distancesUn.append(np.sqrt(np.sum(np.square(emb - new_emb))))
                                                index1 = np.argmin(distancesUn)
                                                conf = 100 - int(distancesUn[index1])
                                                # if(conf>100): conf = 100 
                                                if(distancesUn[index1] < self.config['fcUnCmprVal']):
                                                    idFace = self.unFaceSavedNames[index1]
                                                    if(not idFace in curDetectedIds.keys()): curDetectedIds[idFace] = [self.config['fcFrAdd'],self.config['fcFrDel']]
                                                    if(curDetectedIds[idFace][0] == 0):
                                                        trNewEvent = True 
                                                        events_data.append(['1',idFace,'({}%)'.format(conf),'PhotoSeriesQuene',str(counter)])
                                                    curDetectedIds[idFace] = [curDetectedIds[idFace][0]-1,self.config['fcFrDel']]
                                                    temp_timers["name2"] = round(time.time()-temp_timers["start"],4)
                                                else: # unknown appear the first time - try save data about new person
                                                    faceNameText += 'closest saved unknown face is {} ({}%)'.format(idFace,conf)
                                                    if(not len(unFace_embs)):
                                                        unFace_embs.append(new_emb)
                                                        unFace_cnts.append(self.config['fcFrAdd'])
                                                    else: 
                                                        distances = []
                                                        for emb in unFace_embs: distances.append(np.sqrt(np.sum(np.square(emb - new_emb))))
                                                        index = np.argmin(distances)
                                                        if (distances[index] < self.config['fcUnCmprVal']):
                                                            if(unFace_cnts[index]==0):
                                                                new_name ='unknown_'+str(random.randint(100000, 1000000))
                                                                print("!!!!! add new face "+new_name)
                                                                self.addPhotoSeriesQueue(unFace_embs[index],new_name,face)
                                                                conf = '(100%)'
                                                                del unFace_embs[index]
                                                                del unFace_cnts[index]
                                                                curDetectedIds[new_name] = [self.config['fcFrAdd'],self.config['fcFrDel']]
                                                                trNewEvent = True
                                                                events_data.append(['2',idFace,conf,'PhotoSeriesTemp',str(counter)])
                                                            unFace_cnts[index] -=1
                                                        else: # store unknown face for next frame compare
                                                            unFace_embs.append(new_emb)
                                                            unFace_cnts.append(self.config['fcFrAdd'])
                                        temp_timers["name3"] = round(time.time()-temp_timers["start"],4)
                                        LabelText = "{}:{:n}%".format(idFace, conf)
                                        faceNameText += LabelText
                                        # print(faceNameText)
                                        (text_width, text_height) = cv2.getTextSize(LabelText, self.font, fontScale=0.5, thickness=1)[0]
                                        box2 = (locName[0]+ text_width + 2, locName[1] - text_height + 2)
                                        cv2.rectangle(frame, locName, box2, (5, 5, 5), cv2.FILLED)
                                        cv2.putText(frame, LabelText, locName, self.font, 0.5, (255, 255, 255), 1)
                                        temp_timers["names"] = round(time.time()-temp_timers["start"],4)
                            # save current frame
                            ret2, iWeb = cv2.imencode(self.config['imgFrmt'],frame)
                            if(ret2): self.img = iWeb.tobytes()
                            else: self.log.error("Can't convert img")
                            # check if person not in frame
                            del_names = []
                            for idDel in curDetectedIds.keys(): 
                                curDetectedIds[idDel][1] -=1 
                                if(curDetectedIds[idDel][1]<0): 
                                    del_names.append(idDel)
                                    trNewEvent = True
                            # send alert fow actions
                            if(trNewEvent):
                                saveImgPath = 'static/logs/img/'+str(time.time())+'.jpg'
                                # with open(saveImgPath, "wb") as f: f.write(self.img)  # cv2.imwrite(saveImgPath, frame)
                                evntText = ''
                                for evnt in events_data: evntText += " Appear new persons: in fr"+evnt[0]+' '+evnt[1]+':'+evnt[2]+self.log_divider+'<img src="static/'+evnt[3]+'/' + evnt[1] + '"/>, '
                                for delName in del_names: 
                                    evntText+=" Leave persons: "+delName +" fr "+str(counter)
                                    del curDetectedIds[delName]
                                msg = evntText+self.log_divider+'<img src="' + saveImgPath + '"/>'
                                self.log.warning(msg)
                                self.sendMsgAlert("Camera event", msg, saveImgPath)
                            if(not self.isRun):
                                # self.log.info("Camera:")
                                requests.get(self.serverUrl+'/cameraReady?camId='+self.id+'&type='+self.startMsg)
                                self.isRun = True 
                                self.isStarting = False
                            temp_timers["fps"] = round(1.0/(time.time()-temp_timers["start"]),1)
                            # print(temp_timers)
                            self.timers = temp_timers
                    except Exception as e: 
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback,  limit=2, file=sys.stdout)
                        print('frame error')
                        print(e)
                        # self.log.error("error in takeFrame")
                        # self.exit()
            else: self.exit()
        else: self.log.error("Camera obj is not exist")
    
    def sendMsgAlert(self, subj, text, img_path):
        print("start try send email")
        # if self.toaddr:
        if(False):
            try:
                msg = MIMEMultipart()
                msg['Subject'] = subj
                # me == the sender's email address
                # family = the list of all recipients' email addresses
                msg['From'] = ', '.join(fromaddr)
                msg['To'] = ', '.join(toaddr)
                msg.preamble = 'Multipart massage.\n'
                part = MIMEText(text)
                msg.attach(part)
                # with open("img/Oleg_Sylver_0.png", 'rb') as fp:
                #     img_data = fp.read()
                if(img_path != ""):
                    with open(img_path, 'rb') as fp:
                        img_data = fp.read()
                    part = MIMEApplication(img_data)
                    part.add_header('Content-Disposition', 'attachment', filename="Photo.png")
                    msg.attach(part)
                # print("send mail 1")
                # Send the email via our own SMTP server.
                with smtplib.SMTP_SSL('smtp.gmail.com:465') as s:
                    s.ehlo()
                    s.login(fromaddr, password)
                    s.send_message(msg)
                    # print("send mail")
            except:
                print("Error: unable to send email")

# Camera("testCamera", 0,'').start()