# import glob
# import tensorflow as tf
# from fr_utils import *
# from inception_blocks_v2 import *
# from keras import backend as K

import threading
import cv2
import numpy as np
import time
import face_recognition
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
log = None
log_divider = '##'
class Camera(threading.Thread):
    def __init__(self, id, url, serverUrl, config=None):
        if(id):
            self.isStarting = True
            self.isRun = False
            self.id = id
            self.log = logging.getLogger('Camera_' + id)
            formatter = logging.Formatter('%(asctime)s'+log_divider+'%(levelname)s'+log_divider+'%(message)s')
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
                        'cId': id, 
                        'cName': "Cam",
                        'cUrl': url,
                        'videoScale':1, # scale video output
                        'tryCounter':3, # how many times try connect to a camera 
                        'skipFames':1, # take each N frame from video stream
                        'faceBorder': 10, # border around detected faces in pixels
                        'scaleFactor': 1.05, # 
                        'compareValue': 0.7,
                        'minNeighbors':5,
                        'minSize':30,
                        'isGray': False, # True or False - PICTURE colors or gray

                        'isCompare': True,
                        'isCompareSlow': False,
                        'isDetect': True,
                        'classifierIndex': 0,
                        'imgWebFormat': '.webp',   # '.jpg' .webp .png
                        'imgWebQuality': 95, # 1-99% 
                        'confThreshold': 0.5, # value for TF face detection
                        'facesInMonitoring':0,
                        'facesSkipMonitoring':0,
                        'delayBeforeLeave':5,
                        'delayBeforeAdd':2
                    }
            self.tryCounter = self.config['tryCounter']
            self.cap = None
            self.img = None
            self.timers= {"startCamera":time.time()}
            self.facePhotos = []
            self.faceSources = []
            self.faceEncodes = []
            self.faceNames = []
            self.faceUnknows = []
            self.knownFaceEncodings = []
            self.knownFaceNames = []
            self.font = cv2.FONT_HERSHEY_DUPLEX
            self._stopevent = threading.Event()
            self.isRestart = False
            self.stopDelay = False
            self.slowNames = {}
            self.reloadConfigItems = ['camUrl','isCompare','classifierIndex']
            self.log.info("start camera config ok")
            threading.Thread.__init__(self)
        else: self.log.error("Camera name can not be empty.")
    
    def loadFaces(self):
        location_faces = os.path.join(os.getcwd(), "static/faces/")
        self.log.info("Start loading photos from "+location_faces)
        count = 0
        count_err = 0
        for fileName in os.listdir(location_faces):
            abs_file_path = os.path.join(location_faces, fileName)
            try:
                picture = face_recognition.load_image_file(abs_file_path, mode='RGB')
                encoding = face_recognition.face_encodings(picture)[0]
                self.knownFaceEncodings.append(encoding)
                self.knownFaceNames.append([os.path.splitext(fileName)[0],fileName])
                count += 1
            except:
                count_err += 1
                self.log.error("error open file: <img src=\"faces/" + str(fileName) + "\"/>")
        self.log.info("Knowned faces loaded successefull. Faces: " + str(count) + ". Errors: " + str(count_err))
        self.config['facesInMonitoring'] = count
        self.config['facesSkipMonitoring'] = count_err
    
    def run(self):
        self.log.info("start camera")
        self.isStarting = True
        self.cap = cv2.VideoCapture(self.config['camUrl'])
        if(self.config['isCompare']): self.loadFaces()
        if(self.config['isCompareDNN']):

            for imagePath in sorted(os.listdir("static/PhotoSeries/")):
                id = os.path.split(imagePath)[-1].split(".")[0]
                name = os.path.split(imagePath)[-1].split(".")[1]
                if not id in self.slowNames.keys(): 
                    self.slowNames[id] = [name,imagePath]
                    # print(id,imagePath)
            # print(self.slowNames)
        self.get_frame()
    
    def stopCam(self):
        # self.log.info("pause camera")
        self.stopDelay = False
        self.startMsg = "started"
        self.img = None
        self.isStarting = False
        self.tryCounter = self.config['tryCounter']
        self._stopevent.set()
        self.isRun = False
    
    def exit(self):
        # self.log.info("stop camera")
        self.stopCam()
        if(self.cap):
            if(self.cap.isOpened()): self.cap.release()
        self.cap = None
        self.log.info("camera object is destroyed")
        #cv2.destroyAllWindows()
    
    def get_frame(self):
        if(self.cap):
            cnt = self.config['skipFames']
            # imgWebCompression = []
            # if(self.config['imgWebFormat']=='.png'): imgWebCompression = [cv2.IMWRITE_PNG_COMPRESSION, self.config['imgWebQuality']/10]
            # elif(self.config['imgWebFormat']=='.jpg'): imgWebCompression = [cv2.IMWRITE_JPEG_QUALITY, self.config['imgWebQuality']]
            # else: imgWebCompression = [cv2.IMWRITE_WEBP_QUALITY, self.config['imgWebQuality']]
            # 
            counter = 0
            net = None
            # newDetectedId = {}
            newDetectedIdSlow = {}
            trFaseCascade = False
            print("classifier index="+str(self.config['classifierIndex']))
            if(self.config['classifierIndex']==5):  net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
            else:
                faceCascade = cv2.CascadeClassifier(self.classifiers[self.config['classifierIndex']])
                trFaseCascade = True
            # else: faceCascade = cv2.CascadeClassifier(self.classifiers[self.config['classifierIndex']])
            if(self.config['isCompareDNN']):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read("data/traindata.yml")
            yScale = int(self.config['faceBorder']*3)
            if(not self.stopDelay):
                while not self._stopevent.isSet():
                    try:
                        temp_timers = {"N" : counter,"start" : time.time()}
                        #print(temp_timers)            
                        #try:
                        if(self.cap.isOpened()):
                            cnt -= 1
                            if (cnt > 0): time.sleep(0.1)
                            else:
                                cnt = self.config['skipFames']
                                ret, frame = self.cap.read()
                                if(ret):
                                    trNewEvent = False
                                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    # if(self.config['isGray']): frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    # if(self.config['videoScale']!=1): frame = cv2.resize(frame, (0, 0), fx=self.config['videoScale'], fy=self.config['videoScale'])
                                    if(self.config['isDetect']):
                                        fh, fw = frame.shape[:2]
                                        faceLocations = []
                                        if(trFaseCascade): 
                                            #faceLocations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn") 
                                            faceLocations = faceCascade.detectMultiScale(frame_gray,scaleFactor=self.config['scaleFactor'],minNeighbors=self.config['minNeighbors'],minSize=(self.config['minSize'],self.config['minSize']),flags=cv2.CASCADE_SCALE_IMAGE) #
                                        else:
                                            # cv2.resize(frame, (300, 300))= image, scalefactor, size, mean, swapRB, crop) 
                                            blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (103.93, 116.77, 123.68))
                                            # blob = cv2.dnn.blobFromImage(frame, 0.00392, (300,300), (0,0,0), True, crop=False)
                                            net.setInput(blob)
                                            detections = net.forward()
                                            for i in range(detections.shape[0]):
                                                if (detections[0, 0, i, 2] > self.config['confThreshold']): faceLocations.append((detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int"))
                                        temp_timers["detect"] = round(time.time()-temp_timers["start"],4)
                                        # temp_names = []
                                        temp_names_slow = []
                                        temp_names_slow_conf = []
                                        # temp_faceEncodes = []
                                        # temp_facePhotos = []
                                        # temp_faceSources = []
                                        faceLocationsNames = []
                                        #print(faceLocations)
                                        for (x1, y1, x2, y2) in faceLocations:
                                            # create bigger arrea for a face photo
                                            border = self.config['faceBorder']
                                            bx1=0 if(x1<border) else x1-border
                                            by1=0 if(y1<border) else y1-border
                                            bx2=fw if((x2+border)>fw) else x2+border
                                            by2=fh if((y2+border)>fh) else y2+border
                                            # temp_facePhotos.append(frame[y0:yh, x0:xw])
                                            # temp_faceSources.append(frame_gray[y:y+h, x:x+w])
                                            locName = bx1+2, by1+15
                                            faceLocationsNames.append(locName)
                                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                        # if(len(temp_facePhotos)):
                                            # self.facePhotos = temp_facePhotos[:]
                                            # self.faceSources = temp_faceSources[:]
                                            temp_timers["location"] = round(time.time()-temp_timers["start"],4)
                                            if(self.config['isCompareDNN']):
                                                # print("isCompareSlow")
                                                # print(x1, y1, x2, y2)
                                                face = frame_gray[y1:y2, x1:x2]
                                                # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                                # face = cv2.resize(face,(96,96), interpolation = cv2.INTER_AREA)
                                                idF, cf = recognizer.predict(face) #recognize the Face
                                                # print(idF, cf)
                                                idFace =str(idF) 
                                                conf = str(int((200-cf)/2))+"%"
                                                    # print(self.slowNames[idFace])
                                                faceNameSlow = self.slowNames[idFace][0]+':'+conf
                                                    # print(faceNameSlow)
                                                    # loc =  faceLocationsNames[slowIndex][0],faceLocationsNames[slowIndex][0]+15
                                                    # print(loc)
                                                if cf>140: faceNameSlow += '???' # then less then close to ideal
                                                (text_width, text_height) = cv2.getTextSize(faceNameSlow, self.font, fontScale=0.5, thickness=1)[0]
                                                box2 = (locName[0]+ text_width + 2, locName[1] - text_height + 2)
                                                cv2.rectangle(frame, locName, box2, (5, 5, 5), cv2.FILLED)
                                                cv2.putText(frame, faceNameSlow, locName, self.font, 0.5, (255, 255, 255), 1)
                                                # print('check if already detected')
                                                if(not idFace in newDetectedIdSlow.keys()): 
                                                    # проверить в каком по счету фрейме  появлялся до этого и если он меньше укзаного в конфиге, то выбрасывать алерт
                                                    trNewEvent = True
                                                    temp_names_slow.append(idFace)
                                                    temp_names_slow_conf.append(conf)
                                                    cv2.imwrite('static/logs/img/face_'+str(time.time())+'.jpg',face)
                                                    # print("appear "+idFace)
                                                    # update or add a new person disappear frame counter
                                                # print(newDetectedIdSlow)
                                                newDetectedIdSlow[idFace] = self.config['delayBeforeLeave']
                                                # print(temp_names_slow)
                                            # check if person not in frame
                                    del_names_slow = []
                                    ret2, iWeb = cv2.imencode(".jpg",frame)
                                    if(ret2): self.img = iWeb.tobytes()
                                    else: self.log.error("Can't convert img")
                                    for idDel in newDetectedIdSlow.keys():
                                        if(not idDel in temp_names_slow): 
                                            if(newDetectedIdSlow[idDel]>0): newDetectedIdSlow[idDel] -=1 
                                            else: 
                                                del_names_slow.append(idDel)
                                                trNewEvent = True
                                    if(trNewEvent):
                                        # print('NewEvent')
                                        saveImgPath = 'static/logs/img/'+str(time.time())+'.jpg'
                                        with open(saveImgPath, "wb") as f: f.write(self.img)  # cv2.imwrite(saveImgPath, frame)
                                        evntText = 'Detected: '
                                        for i2 in range(len(temp_names_slow)):
                                            print(counter,"appear")
                                            evntText += "Appear new persons:"
                                            if(i2 != 'unknown'): evntText += self.slowNames[temp_names_slow[i2]][0]+':'+temp_names_slow_conf[i2]+log_divider+'<img src="static/PhotoSeries/' + self.slowNames[temp_names_slow[i2]][1] + '"/>, '
                                            else: evntText += self.slowNames[temp_names_slow[i2]][0]+':'+temp_names_slow_conf[i2]+log_divider+ self.slowNames[temp_names_slow[i2]][1] + '"/>, '
                                        for delName in del_names_slow: 
                                            evntText+="Leave persons: "+self.slowNames[delName][0]
                                            print(counter,"disappear")
                                            del newDetectedIdSlow[delName]
                                        self.log.warning(evntText+log_divider+'<img src="' + saveImgPath + '"/>')
                                        # print(evntText)
                                    temp_timers["names"] = round(time.time()-temp_timers["start"],4)
                                    counter += 1
                                    if(not self.isRun):
                                        # self.log.info("Camera:")
                                        requests.get(self.serverUrl+'/cameraReady?camId='+self.id+'&type='+self.startMsg)
                                        self.isRun = True 
                                        self.isStarting = False
                                    temp_timers["fps"] = round(1.0/(time.time()-temp_timers["start"]),1)
                                    # print(temp_timers)
                                    self.timers = temp_timers
                        else: print("skip frame")
                    except Exception as e: 
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback,  limit=2, file=sys.stdout)
                        print('frame error')
                        print(e)
                        # self.log.error("error in takeFrame")
                        # self.exit()
            else: self.exit()
        else: self.log.error("Camera obj is not exist")

# Camera("testCamera", 0,'').start()