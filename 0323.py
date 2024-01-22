#0323.py
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

train_faces, train_labels, test_faces, test_labels = load_face()

#2    
recognizer = cv2.face.LBPHFaceRecognizer_create()
##recognizer.train(train_faces.reshape(-1, HEIGHT, WIDTH), train_labels)
##recognizer.save('./data/LBPH_face_train.yml')

#3: predict test_faces using recognizer
recognizer.read('./data/LBPH_face_train.yml')
correct_count = 0
for i, face in enumerate(test_faces.reshape(-1, HEIGHT, WIDTH)):    
    predict_label, confidence = recognizer.predict(face)
    if test_labels[i]== predict_label:
        correct_count+= 1
    #print('test_labels={}: predicted:{}, confidence={}'.format(
    #                 test_labels[i], predict_label,confidence))
accuracy = correct_count / len(test_faces)
print('test_faces, accuracy=', accuracy)
