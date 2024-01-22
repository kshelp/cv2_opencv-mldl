#0402.py
import tensorflow as tf
import numpy as np

#1: load saved model in 0401.py

#1-1: SavedModel
model = tf.keras.models.load_model("./dnn/SAVED_MODEL") 

#1-2: Checkpoint with save_weights_only=False
#model = tf.keras.models.load_model("./dnn/ckpt/0401-0100.ckpt") 

model.summary()

#2
X = np.array([[0, 0], [0, 1], [1, 0],  [1, 1]], dtype = np.float32)

outs = model(X)
pred = np.argmax(outs, axis = 1)
print('pred=', pred)

#3
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "XOR2.pb")
