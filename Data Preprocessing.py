import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle

print("Packages installed")

# img=cv2.imread("dataset/with_mask/00000_Mask.jpg")
# img=cv2.resize(img,(224,224))
# cv2.imshow("Image",img)
# cv2.waitKey(0)

Datadirectory="dataset/"
CATEGORIES=["with_mask","without_mask"]
training_data=[]
for category in CATEGORIES:
    path=os.path.join(Datadirectory,category)
    Label=CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (224, 224))
            training_data.append([new_array, Label])
        except Exception as e:
            pass

print(len(training_data))

random.shuffle(training_data)
training_data=training_data[::-1]
random.shuffle(training_data)

X=[]   #data
y=[]   #label
for features,label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,224,224,3)
print("X shape: ",X.shape)

#Normalizing the data
X=X/225.0
print("val Y",y[1000])

Y=np.array(y)
print(Y.size)

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

