import sys
import cv2
import os
import time
import numpy as np
# from scipy import misc
import tensorflow as tf
from tensorflow.python.platform import gfile
import detect_face
import pickle
# import math
# import imageio 
# import argparse

import re

# python3 classifier.py  TRAIN static/PhotoSeries/ data/20180402-114759/20180402-114759.pb data/myClassifier.pkl

# Object arrays cannot be loaded when allow_pickle=False
# Look inside your env's site-packages directory: .../site-packages/numpy/lib/npio.py
# In line 292: def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, ...)
# Change allow_pickle attribute from False to True and everything gonna work again.
class PreProcessor():
    def __init__(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default(): self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
            self.minsize = 20 # minimum size of face
            self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            self.factor = 0.709 # scale factor

    def align(self, img, margin=44, image_size=160):
        # img = misc.imread(image_path)
        img = img[:,:,0:3]
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        bb = np.zeros(4, dtype=np.int32)
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det = (det[index,:])
        else: return bb
        det = np.squeeze(det)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        # scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        # misc.imsave("temp.png", scaled)
        return bb

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
        classifier_filename_exp = os.path.expanduser("data/myClassifier.pkl")
        with open(classifier_filename_exp, 'rb') as infile: (self.model, self.class_names) = pickle.load(infile)

    def __del__(self):
        self.sess.close()

    def predict(self, face):
        image = self.load_image(face)
        self.feed_dict = { self.images_placeholder:image, self.phase_train_placeholder:False }
        self.emb_array[0,:] = self.sess.run(self.embeddings, feed_dict=self.feed_dict)
        predictions = self.model.predict_proba(self.emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        return self.class_names[best_class_indices[0]],int(best_class_probabilities[0]*100), self.emb_array

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

    def load_model(self, model="data/20180402-114759/20180402-114759.pb"):
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
        if len(meta_files)==0: raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1: raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
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

required_size=(160, 160)
dnnDatect =True
if(dnnDatect): net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")
else: preprocessor = PreProcessor()

classifier = Classify()
camera = cv2.VideoCapture(0)

while True:
    start = time.time()
    ret, frame = camera.read()
    bb = None
    name = ''
    conf = ''
    if(dnnDatect):
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (103.93, 116.77, 123.68))
        net.setInput(blob)
        fh, fw = frame.shape[:2]
        detections = net.forward()
        txt =''
        for i in range(detections.shape[0]):
            if (detections[0, 0, i, 2] > 0.5):
                x1,y1,x2,y2 = (detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int")
                face = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2],required_size), cv2.COLOR_BGR2RGB)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                idFace, conf, new_emb = classifier.predict(face)
    else:
        bb = preprocessor.align(frame)
        cv2.rectangle(frame, (bb[0],bb[1]), (bb[2],bb[3]), (0, 255, 0), 2)
        if(len(bb) and bb[0]!=0):
            face = cv2.cvtColor(cv2.resize(frame[bb[1]:bb[3],bb[0]:bb[2]],required_size), cv2.COLOR_BGR2RGB)
            # face = face.astype('float32')
            idFace, conf, new_emb = classifier.predict(face)
        # print(idFace)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fps = str(int(1/(time.time()-start)))
    # print(fps, idFace, conf)
    cv2.putText(frame, "fps={} name={}({}%)".format(fps,idFace,conf), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.4, (5, 5, 250), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()
# print(classifier.predict('temp.png'))