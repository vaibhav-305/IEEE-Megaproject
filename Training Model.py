import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle

model=tf.keras.applications.mobilenet.MobileNet()
#model.summary()

#Tuning Weights

base_input=model.layers[0].input
base_output=model.layers[-4].output

Flat_layer=layers.Flatten()(base_output)
final_output=layers.Dense(1)(Flat_layer)
final_output=layers.Activation('sigmoid')(final_output)

new_model=keras.Model(inputs=base_input,outputs=final_output)
#new_model.summary()

#Setting for binary classification (Face mask/without mask)

# X=np.load('Data.npy')
# Y=np.load('Label.npy')
pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)
Y=np.array(y)

print("Everything in")

opt = keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
new_model.fit(X,Y,epochs=3,validation_split=0.1)
new_model.save('Mask_detector.model',save_format="h5")