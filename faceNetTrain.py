# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
import time
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

folderPathVal = "static/PhotoSeriesVal/"
folderPathTrain = "static/PhotoSeries/"

net = cv2.dnn.readNetFromTensorflow("data/opencv_face_detector_uint8.pb", "data/opencv_face_detector.pbtxt")

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    face_array = []
    try:
        # image = Image.open(filename).convert('RGB')
        image = cv2.imread(filename)
        fh, fw = image.shape[:2]
        face_array = []
        blob = cv2.dnn.blobFromImage(image, 1, (300,300), (103.93, 116.77, 123.68))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[0]):
                if (detections[0, 0, i, 2] > 0.6): 
                    x1,y1,x2,y2 = (detections[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype("int")
                    face_array = image[y1:y2, x1:x2]
                    # resize pixels to the model size
                    face_array = cv2.resize(face_array,required_size)
                    face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
    except: print("Skip file {}".format(filename))
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        if(len(face)): faces.append(face)
        else: 
            print("error add file {}".format(path))
            break
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path): continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

print("start")
trainX, trainy = load_dataset(folderPathTrain)
print(trainX.shape, trainy.shape)
testX, testy = load_dataset(folderPathVal)
print(testX.shape, testy.shape)
np.savez_compressed('data/faces-dataset.npz', trainX, trainy, testX, testy)
print("created a dataset")

# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32') #!!!!!!!!!!!!!
    face_pixels = (face_pixels - face_pixels.mean()) / face_pixels.std()
    samples = np.expand_dims(face_pixels, axis=0)
    face = model.predict(samples)
    return face[0]

# load the face dataset
# dataset = np.load('data/faces-dataset.npz')
# trainX, trainy, testX, testy = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3']
# print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
modelFaceNet = load_model('data/facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX: newTrainX.append(get_embedding(modelFaceNet, face_pixels))
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX: newTestX.append(get_embedding(modelFaceNet, face_pixels))
newTestX = np.asarray(newTestX)
print(newTestX.shape)
np.savez_compressed('data/faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
print("created an embedding store")

# load dataset
# data = np.load('data/faces-embeddings.npz')
# newTrainX, trainy, testX, testy = dataEmbeding['arr_0'], dataEmbeding['arr_1'], dataEmbeding['arr_2'], dataEmbeding['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.2f, test=%.2f' % (score_train*100, score_test*100))