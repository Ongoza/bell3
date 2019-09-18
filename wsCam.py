# pip install -r requirements.txt
import signal
import threading
import time
#import json
import os
import sys
from subprocess import check_output
import time
import random
import tornado.gen
from datetime import date
import camera2 as camera
import cv2
import json

import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template
import tornado.options
import tornado.autoreload
from loguru import logger
from tornado.options import options, define
# async request
from tornado.gen import coroutine

# json format Ctrl+Shift+I 
define("host", default="localhost", help="app host", type=str)
define("port", default=9090, help="app port", type=int)

serverUrl = "http://"+options.host+":"+str(options.port)
cameraIds = ['USB0']
clients = []
streamTimers = []
#cameras = []
cam = None
curFrame=0
root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir,"data")
#camera = "rtsp://admin:12345qwer@192.168.1.202"
cameraURL = 0 # camera address

# logger.add("static/logs/server.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
# logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.debug("Start server in debug mode")

# Create target Directory if don't exist
if not os.path.exists(data_dir):
    try:
        os.mkdir(data_dir)
        #logger.debug("Directory " , os.path.abspath(data_dir) ,  " Created ")
    except Exception as e:
        logger.error("Error create Directory " , os.path.abspath(data_dir))
else:
    logger.debug("Directory " , os.path.abspath(data_dir) ,  " Exist")

executor = tornado.concurrent.futures.ThreadPoolExecutor()
class AsyncHandler(tornado.web.RequestHandler):
    @coroutine
    def get(self):
        ans = yield executor.submit(self.do_slow)
        self.write(ans)

    def do_slow(self):
        time.sleep(2)
        a = 'okww2'
        return a

class MainHandler(tornado.web.RequestHandler):
  def get(self):
    #loader = tornado.template.Loader(".")
    #self.write(loader.load("test.html").generate())
    self.render(os.path.join("static","index.html"), host=options.host, port=options.port)

class LogHandler(tornado.web.RequestHandler):
  def get(self):
    #loader = tornado.template.Loader(".")
    #self.write(loader.load("test.html").generate())
    self.render(os.path.join("static","logCam.html"), host=options.host, port=options.port)

class GetLogHandler(tornado.web.RequestHandler):
  def get(self):
    cnt = self.get_argument('count', 2)
    fName = self.get_argument('name', 'camera_USB0.log')
    print(self)
    answ = ''
    print(cnt)
    #loader = tornado.template.Loader(".")
    answ = check_output(['tail', '-'+str(cnt), 'static/logs/'+fName])
    self.write(answ)


class WsCameraHandler(tornado.websocket.WebSocketHandler):
  def open(self):
    logger.info("cam is ready for web stream")
    if(clients[0]): clients[0].write_message({'0':'log','1':'Camera object open for stream','2':cam.id})

  def on_message(self, message):
    if(cam.img is not None): 
        # print(cam.img)
        self.write_message(cam.img,True)
    else:
        logger.error("skip frame in stream")
        if(clients[0]): clients[0].write_message({'0':'error','1':'cameraClosed','2':cam.id})

  def on_close(self):
    logger.info('camera connection closed...')

# def wsBroadcast ():
#   for client in clients:
#       client.send("data")
# [I 190812 00:23:29 camera:108] destroy camera object
# select timeout
# VIDIOC_DQBUF: Resource temporarily unavailable
# sudo apt-get install linux-generic-lts-utopic

class WsCmdHandler(tornado.websocket.WebSocketHandler):  
  def open(self):
    logger.info("start command connection")
    clients.append(self)
    #self.write_message("The server says: 'Hello'. Connection was accepted.")

  def on_message(self, message):
    global cam
    data = json.loads(message)
    # print(data)
    if(data['0']=='takeTimers'):
        # print(data)
        if(cam): self.write_message({'0':"timers",'1':cam.timers}) if(cam.isRun) else self.write_message({'0':'error','1':'cameraClosed','2':cam.id})
        else: self.write_message({'0':'error','1':'cameraClosed','2':'cam.id'})
    elif(data['0']=='camConfTake'):
        if(cam): self.write_message({'0':"config",'1':cam.config,'2':cam.id})
        else: self.write_message({'0':'error','1':'cameraClosed','2':'cam.id'})
    elif(data['0']=='camConfSave'):
        if(data['1'] and data['2']):
            # print(data['2']) 
            with open('configs/'+data['1']+'.json', 'w+') as outfile:
                json.dump(data['2'], outfile)
            self.write_message({'0':'log','1':'Camera config saved successfull','2':cam.id})
        else: self.write_message({'0':'error','1':'can not save config','2':cam.id})
    elif(data['0']=='camConfApply'):
        print('camConfApply')
        try:
            if(cam): 
                config = data['2']
                reloadItems = []
                oldId = cam.id
                for value in config:
                    if(config[value] != cam.config[value]):
                        if(value in cam.reloadConfigItems):
                            reloadItems.append(value)
                # print(reloadItems)
                # cam.config.update(config)
                if(len(reloadItems)>0): 
                    if(cam != None): 
                        cam.exit()
                        cam = None
                        # print(config['camId'],config['camUrl'],serverUrl)
                        cam = camera.Camera(config['cId'],config['cUrl'],serverUrl,config)
                        # print('start')
                        cam.start()
                    # cam.stopCam()
                    # if('camUrl' in reloadItems): cam.cap = cv2.VideoCapture(cam.config['camUrl'])
                    # elif('isEncode' in reloadItems): cam.loadFaces()
                    # cam.startMsg = "reloaded and restarted"
                    # print('self.get_frame()')
                    # cam.get_frame()
                else:self.write_message({'0':'system','1':'confApplied','2':cam.id,'3':config})
            else: self.write_message({'0':'error','1':'cameraNotConnected','2':'oldId'})
        except Exception as e: 
            print(e)
            self.write_message({0:"error",1:'Can not aplly this configuration','2':oldId,'3':config})
        print("exit apply")
    elif(data['0']=='camStart'):
        if(cam): 
            self.write_message({'0':'log','1':'Camera object exist','2':cam.id})
            if(cam.isRun or cam.isStarting): 
                self.write_message({'0':'log','':'Camera object is running','2':cam.id})
                self.write_message({'0':"config",'1':cam.config,'2':cam.id})
            else:
                self.write_message({'0':'log','1':'Camera object starting','2':cam.id})
                cam.start()
        else:
            if(cam == None): 
                cam = camera.Camera("USB0",0,"http://"+options.host+":"+str(options.port))
                self.write_message({'0':'log','1':'Camera object created','2':cam.id})
            if(cam): 
                if(not cam.isRun or not cam.isStarting):
                    self.write_message({'0':'log','1':'Camera object starting','2':cam.id})
                    cam.start()
                else:
                    self.write_message({'0':'log','1':'Camera object already starting','2':cam.id})
                    logger.info("can not run cam.start()")
            else: 
                logger.error("Can not create camera object")
                self.write_message({'0':'log','1':'Can not create camera object','2':'cam.id'})
    elif(data['0']=='camStop'):
        logger.info(data)
        typeStop = 0
        if(cam): 
            if(cam.isRun):
                cam.exit()
                typeStop = 1
            else: 
                cam.stopDelay = True
                typeStop = 2
        self.write_message({'0':'system','1':'cameraStopOk','2':cam.id,'3':typeStop})
        cam = None
    else: self.write_message({'0':'error','1':'unKnowCommand','2':cam.id})

  def on_close(self):
    logger.info('cammand connection closed...')
    clients.remove(self)

class CameraReadyHandler(tornado.web.RequestHandler):
    # http://localhost:9090/cameraReady?camId=www&type=ddd
    def get(self):
        print(self.path_args)
        cId = self.get_argument('camId', '')
        strType = self.get_argument('type', '')
        # if(strType == 'confApplied'):
        #     for client in clients:
        #         client.write_message({'0':'log','1':'confUpdated','2':cId})
        # else:
        for client in clients:
            client.write_message({'0':'system','1':'takeConfig','2':cId})
            client.write_message({'0':'log','1':'Camera: '+strType,'2':cId})

class NotFoundHandler(tornado.web.RequestHandler):
	def get(self):
		logger.debug("Not founded page "+self.request.body)
		self.write('{"error":"404"}')

class MyApplication(tornado.web.Application):
    is_closing = False

    def signal_handler(self, signum, frame):
        logger.debug("\nexiting..."+str(self.is_closing) )
        self.is_closing = True
    
    def try_exit(self):
        global cam
        # logger.debug("\nexiting 2..."+str(self.is_closing) )
        if self.is_closing:
            try:
                logger.debug("start exit")
                if(cam != None): 
                    cam.exit()
                    cam.join()
                time.sleep(3)
                cam = None
                tornado.ioloop.IOLoop.instance().stop()
                logger.debug("exit ok")
                #print(time.strftime('%Y-%m-%d %H:%M:%S '))
            except Exception as e: 
                print(e)
                logger.error("Stop function has some troubles")

application = MyApplication([
        (r'/', MainHandler),
        (r"/cmd", WsCmdHandler),
        (r"/log", LogHandler),
        (r"/getlog", GetLogHandler),
        # (r"/cameraReady", CameraReadyHandler),  ?P<param1>[^\/]+)/?(?P<param2>[^\/]+)?/?(?P<param3>[^\/]+)?" r"/users/key=(?P<key>\w+)"
        (r"/cameraReady", CameraReadyHandler),
        (r"/cam", WsCameraHandler),
        (r"/static/(.*)",tornado.web.StaticFileHandler, {"path": "static"},),
        # (r"/images/(.*)",tornado.web.StaticFileHandler, {"path": "static/images"},),
        # (r"/js/(.*)",tornado.web.StaticFileHandler, {"path": "static/js"},)
        ],
	    debug=True,
	    static_hash_cache=False
    )


if __name__ == "__main__":
    listDirs = ['configs/','static/','data/', 'static/logs/', 'static/js/', 'static/images/','static/PhotoSeriesTemp','static/PhotoSeriesQueue','static/PhotoSeries']
    for directory in listDirs:
        if not os.path.exists(directory):
            print("create new dir " + directory)
            os.makedirs(directory)
    tornado.options.parse_command_line()    
    signal.signal(signal.SIGINT, application.signal_handler)
    #cam = camera.Camera("USB0",cameraURL,"USB0",)
    #cam.start()
    try:
        application.listen(options.port)
        logger.info("start websocketServer on port: "+str(options.port))
        tornado.ioloop.PeriodicCallback(application.try_exit, 100).start()
        logger.info("Press Ctrl-C for stop the server.")
        tornado.autoreload.watch(os.path.join(root_dir,"wsCam.py"))
        tornado.autoreload.watch(os.path.join(root_dir,"index.html"))
        for dir, _, files in os.walk('static/js'):
            for f in files:                
                if not f.startswith('.'):
                    #logger.debug(dir + '/' +f)
                    tornado.autoreload.watch(dir + '/' + f)
        tornado.ioloop.IOLoop.instance().start()        	
    except Exception as e:
        logger.error(e)