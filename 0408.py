#0408.py
'''
ref1: https://keras.io/examples/vision/image_classification_from_scratch/
ref2: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
'''
import tensorflow as tf
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense 
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.layers   import Rescaling, RandomFlip, RandomRotation 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing  import image_dataset_from_directory

#1: remove corrupted images that do not have "JFIF" in their header and size 0.
##from PIL import Image
##import os, glob
##mode2bpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32,
##            'CMYK':32, 'YCbCr':24, 'I':32, 'F':32}
##files = glob.glob('./PetImages/Cat/*.jpg')
##files.extend(glob.glob('./PetImages/Dog/*.jpg'))
##for file in files:
##    get_size = os.path.getsize(file)
##    if get_size == 0:
##        print(f'{file} : size={get_size}')
##        os.remove(file)
##    try:
##        fobj = open(file, "rb")
##        is_jfif = b'JFIF' in fobj.peek(10)
##    finally:
##        fobj.close()
##        
##    if not is_jfif:
##        print("not JFIF:", file)
##        os.remove(file)
##     
##    img = Image.open(file)
##    bpp = mode2bpp[img.mode]
##    if bpp != 24:
##        img =img.convert(mode="RGB")
##        img.save(file)

#2
#2-1
H, W, C = 224, 224, 3
batch_size = 64
#DATA_PATH = './cats_and_dogs_filtered/train'
DATA_PATH = './PetImages'
train_ds = image_dataset_from_directory(
    directory= DATA_PATH,
    labels = 'inferred',
    label_mode ='categorical',  # 'int', 'binary'
    color_mode = 'rgb',         # 'grayscale'
    batch_size = batch_size,
    image_size=(H, W),
    #shuffle = True,
    seed = 777,
    validation_split=0.2,
    subset="training")

#2-2
val_ds = image_dataset_from_directory(
    directory= DATA_PATH,
    labels = 'inferred',
    label_mode = 'categorical', # 'int', 'binary'
    color_mode = 'rgb',         # 'grayscale'
    batch_size = batch_size,
    image_size=(H, W),
    #shuffle = True,
    seed = 777,
    validation_split = 0.2,
    subset='validation')

#2-3
train_ds = train_ds.prefetch(buffer_size=64)
val_ds   = val_ds.prefetch(buffer_size=64)

#3
#3-1
##import matplotlib.pyplot as plt
##plt.figure(figsize=(10, 10))
##for images, labels in train_ds.take(1):
##    for i in range(16):
##        ax = plt.subplot(4, 4, i + 1)
##        plt.imshow(images[i].numpy().astype("uint8"))
##        plt.title(labels[i].numpy().argmax())
##        plt.axis("off")
##plt.show()

#3-2
##normalize = Rescaling(scale=1./127.5, offset = -1.0)  #[-1, 1]
##train_ds = train_ds.map(lambda x, y: (normalize(x), y))
##val_ds   = val_ds.map(lambda x, y: (normalize(x), y))

#4-1
augmentation = tf.keras.Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.1)])

#4-2
def make_model(input_shape, num_classes): # functional API
    inputs = Input(shape=input_shape)
    x= augmentation(inputs)
    x= Rescaling(scale=1./127.5, offset = -1.0)(x)
    x= Conv2D(filters=16, kernel_size = (3,3), activation='relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout( rate=0.2)(x)
    x= Flatten()(x)
    outputs= Dense(units=2, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)
    
model = make_model(input_shape=(H, W, C), num_classes=2)    
model.summary()    
        
#5
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(train_ds, epochs=30, validation_data=val_ds, verbose=2)

train_loss, train_acc  = model.evaluate(train_ds, verbose=2) 
val_loss,  val_acc  = model.evaluate(val_ds, verbose=2)

#6: freezing
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "Petimages.pb")
