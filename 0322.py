#0322.py

#pip install opencv-contrib-python

import cv2
import numpy as np

#1
WIDTH = 92
HEIGHT = 112
def load_face(filename='./data/faces.csv', test_ratio=0.2, shuffle=True):
    file = open(filename, 'r')
    lines = file.readlines()

    N = len(lines)
    faces = np.empty((N, WIDTH*HEIGHT), dtype=np.uint8 )
    labels = np.empty(N, dtype = np.int32)
    for i, line in enumerate(lines):
        filename, label = line.strip().split(';')
        labels[i] = int(label)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        faces[i, :] = img.flatten()
  
    if shuffle:
        indices = list(range(N))
        np.random.seed(111) # same random sequences, so the same result
        np.random.shuffle(indices)

        faces = faces[indices]
        labels = labels[indices]

    # seperate train and test data
    test_size = int(test_ratio*N)
    test_faces = faces[:test_size]
    test_labels = labels[:test_size]

    train_faces = faces[test_size:]
    train_labels = labels[test_size:]
    return train_faces, train_labels, test_faces, test_labels

#2
train_faces, train_labels, test_faces, test_labels = load_face()
##print('train_faces.shape=',  train_faces.shape) #train_faces.shape= (320, 10304)
##print('train_labels.shape=', train_labels.shape)#train_labels.shape=(320,)
##print('test_faces.shape=',   test_faces.shape)  #test_faces.shape=  (80, 10304)
##print('test_labels.shape=',  test_labels.shape) #test_labels.shape= (80,)

#3: select recognizer_type
EIGEN_FACE, FISHER_FACE = 0, 1
recognizer_type = 1#EIGEN_FACE # FISHER_FACE

#4: train recognizer
#4-1
if recognizer_type == EIGEN_FACE:
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(train_faces.reshape(-1, HEIGHT, WIDTH), train_labels)
    recognizer.save('./data/eigen_face_train.yml')
    
#4-2
else: #FISHER_FACE
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.train(train_faces.reshape(-1, HEIGHT, WIDTH), train_labels)
    recognizer.save('./data/Fisher_face_train.yml')

#5:    
#5-1
if recognizer_type == EIGEN_FACE:
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read('./data/eigen_face_train.yml')
#5-2
else:
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.read('./data/Fisher_face_train.yml')

#6: predict test_faces using recognizer
correct_count = 0
for i, face in enumerate(test_faces): 
    predict_label, confidence = recognizer.predict(face)
    if test_labels[i]== predict_label:
        correct_count+= 1
    #print('test_labels={}: predicted:{}, confidence={}'.format(
    #                 test_labels[i], predict_label,confidence))
accuracy = correct_count / len(test_faces)
print('test_faces, accuracy=', accuracy)


#7: display eigen Face
eigenFace = recognizer.getEigenVectors()
eigenFace = eigenFace.T
print('eigenFace.shape=',  eigenFace.shape)

if recognizer_type == EIGEN_FACE:
    nFace = 80
    dst = np.zeros((8*HEIGHT, 10*WIDTH), dtype=np.uint8)
else: #FISHER_FACE
    nFace = 39
    dst = np.zeros((4*HEIGHT, 10*WIDTH), dtype=np.uint8)
    
for i in range(nFace):  
  x = i%10
  y = i//10
  x1 = x*WIDTH
  y1 = y*HEIGHT
  x2 = x1+WIDTH
  y2 = y1+HEIGHT  
  
  img = eigenFace[i].reshape(HEIGHT, WIDTH)
  dst[y1:y2, x1:x2] = cv2.normalize(img,None ,0,255,cv2.NORM_MINMAX)
cv2.imshow('eigenFace', dst)

cv2.waitKey()
cv2.destroyAllWindows()
