import numpy as np
import os
from PIL import Image
import cv2

path = "../cafffe_lenet/fer2013/dataset/train/"
labelsDict = dict() # label字典
for _,dirs,_ in os.walk(path):
    for d in dirs:
        for _,_,files in os.walk(path+d):
            for f in files:
                labelsDict.update({f:d})

imgs = list()
labels = list()

for _,_,files in os.walk("foolimage"):
    for f in files:
        img = Image.open("foolimage/"+f)
        img = np.array(img.getdata(),dtype='uint8')[:,0]
        img = np.reshape(img,(224,224))
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        img = np.resize(img,(2304))
        imgs.append(img)
        labels.append(labelsDict[f])
imgs = np.array(imgs)
labels = np.array(labels)
np.save("foolL.npy",labels)
np.save("fool.npy",imgs)
print(imgs.shape)