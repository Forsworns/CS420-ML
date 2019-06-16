from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization as BN, MaxPooling2D as MP
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.backend import squeeze
import numpy as np
import tensorflow as tf 
import pandas as pd
import os
import cv2
# from keras.utils.vis_utils import plot_model
from configs import *


inceptionV3pre = 'InceptionV3pre'

class Inception(object):
    def __init__(self,configs):
        self.name = configs.Inception
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.trainable = configs.trainable
        self.dropout = configs.dropout
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.weight_name = 'model/{}-{}-{}-{}.hdf5'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.history_name = 'history/{}-{}-{}-{}.csv'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.prebuilt(self.name)

    def prebuilt(self,name):
        # fine tune 使用imageNet训练过的模型，需要输入rgb图片
        base = None
        if name == inceptionV3pre:
            base = InceptionV3(weights='./model/InceptionV3_ImageNet.h5',include_top=False,input_shape=(224,224,3))
        else:
            base = InceptionV3(weights=None,include_top=False,input_shape=(224,224,3))
        trainable = Sequential()
        trainable.add(Flatten(input_shape=base.output_shape[1:]))
        trainable.add(Dense(256,activation='relu'))
        trainable.add(Dense(7,activation='softmax'))
        print(base.output)
        print(squeeze(base.output,0))
        self.model = Model(inputs=base.input,output=trainable(base.output))
        if not self.trainable:
            for l in range(len(self.model.layers)-1):
                self.model.layers[l].trainable = False

    def train(self,data,label,validation):
        sgd=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.weight_name, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
        history = self.model.fit(data,label,epochs=self.epochs,batch_size=self.batch_size,validation_data=validation,shuffle=True,callbacks=[checkpoint])
        pd.DataFrame.from_dict(history.history).to_csv(self.history_name, float_format="%.5f", index=False)

    def test(self,data,label):
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        loss_and_metric = self.model.evaluate(data,label,batch_size=32)
        print("{} on test set".format(self.name),loss_and_metric)

    def load(self):
        self.model.load_weights(self.weight_name)

    def print(self):
        self.model.summary()
        # plot_model(self.model, to_file="{}.png".format(self.name), show_shapes=True)

flags = tf.app.flags
flags.DEFINE_string('Inception',"InceptionV3","decide the Inception version")
flags.DEFINE_integer('epochs',10,"epochs to train")
flags.DEFINE_integer('batch_size',4,"batch size to train")
flags.DEFINE_float('dropout',0,"dropout rate for full connected layer")
flags.DEFINE_boolean('is_train',True,"train or test")
flags.DEFINE_boolean('trainable',False,"trainable or not")
flags.DEFINE_boolean('fool',True,"fool or test")
configs = flags.FLAGS

def main(_):
    ## data
    train_x = np.load(TD)
    train_y = np.load(TL)  
    if configs.fool:
        fool_x = np.load(FD)
        train_x = np.vstack((train_x,fool_x))
        fool_y = np.load(FL)
        train_y = np.concatenate((train_y,fool_y))    
    train_x = train_x.reshape((-1,48,48)) # 将每行的图片数据reshape成图片 (nx48x48)
    train_x = train_x[:,:,:,np.newaxis] # 将最后一维拓展成channel (nx48x48x1)             
    train_y = to_categorical(train_y) # 转变为类别编码(nx7)

    validate_x = np.load(VD)
    validate_x = validate_x.reshape((-1,48,48))
    validate_x = validate_x[:,:,:,np.newaxis]
    validate_y = np.load(VL)
    validate_y = to_categorical(validate_y)

    test_x = np.load(PD)
    test_x = test_x.reshape((-1,48,48))
    test_x = test_x[:,:,:,np.newaxis]
    test_y = np.load(PL)
    test_y = to_categorical(test_y)

    # 如果直接调用keras提供的resnet，需要将图片拓展到rb三个channel，这里直接复制三次 
    train_x = np.array([train_x,train_x,train_x]).squeeze().transpose((1,2,3,0)) 
    validate_x = np.array([validate_x,validate_x,validate_x]).squeeze().transpose((1,2,3,0)) 
    test_x = np.array([test_x,test_x,test_x]).squeeze().transpose((1,2,3,0)) 
    print(train_x.shape)

    # reshape to 224x224
    train_x_l = np.zeros((train_x.shape[0],224,224,3))
    validate_x_l = np.zeros((validate_x.shape[0],224,224,3))
    test_x_l = np.zeros((test_x.shape[0],224,224,3))
    for n in range(train_x.shape[0]):
        train_x_l[n,...] = cv2.resize(np.array(train_x[n,...].squeeze(),dtype='uint8'), (224, 224), interpolation=cv2.INTER_CUBIC)
    for n in range(validate_x.shape[0]):    
        validate_x_l[n,...] = cv2.resize(np.array(validate_x[n,...].squeeze(),dtype='uint8'), (224, 224), interpolation=cv2.INTER_CUBIC)
    for n in range(test_x.shape[0]):
        test_x_l[n,...] = cv2.resize(np.array(test_x[n,...].squeeze(),dtype='uint8'), (224, 224), interpolation=cv2.INTER_CUBIC)
    
    ## model
    resnet = Inception(configs)
    resnet.print()
    if configs.is_train:
        resnet.train(train_x_l,train_y,(validate_x_l,validate_y))
        resnet.test(test_x_l,test_y)
    else:
        resnet.load()
        resnet.test(test_x_l,test_y)

if __name__ == "__main__":
    tf.app.run()