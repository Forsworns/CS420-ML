from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization as BN, MaxPooling2D as MP
from keras.applications import VGG16, VGG19
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.backend import squeeze
import numpy as np
import tensorflow as tf
import pandas as pd
import os
# import matplotlib.pyplot as plt 

TD = "train_data.npy"
TL = "train_label.npy"
VD = "private_data.npy" # validation
VL = "private_label.npy"
PD = "public_data.npy"
PL = "public_label.npy"

vgg16pre = 'VGG16pre'
vgg19pre = 'VGG19pre'
VGG_Config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(object):
    def __init__(self,configs):
        self.name = configs.VGG
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.weight_name = 'model/{}-{}-{}-{}.hdf5'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.history_name = 'history/{}-{}-{}-{}.csv'.format(self.name,self.epochs,self.batch_size,self.dropout)
        if self.name in VGG_Config.keys():
            self.model = Sequential()
            self.build_model(VGG_Config[self.name])
        else:
            self.model = None
            self.prebuilt(self.name)

    def build_model(self,vgg_config):
        # 从零建立模型，直接处理灰度图像
        first = True
        for channel in vgg_config:
            if channel == 'M':
                self.model.add(MP(strides=2))
            else:
                if first:
                    self.model.add(Conv2D(channel,3,strides=1,input_shape=(48,48,1),padding='same',kernel_initializer='uniform'))
                    self.model.add(BN())
                    self.model.add(Activation('relu'))
                    first = False
                else:
                    self.model.add(Conv2D(channel,3,strides=1,padding='same',activation='relu',kernel_initializer='uniform'))
                    self.model.add(BN())
                    self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(4096,activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(2048,activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(7,activation='softmax'))

    def prebuilt(self,name):
        # fine tune 使用imageNet训练过的模型，需要输入rgb图片
        base = None
        if name == vgg16pre:
            base = VGG16(weights='imagenet',include_top=False,input_shape=(48,48,3))
        else:
            base = VGG19(weights='imagenet',include_top=False,input_shape=(48,48,3))
        base.trainable = False
        transfer = Sequential()
        transfer.add(Flatten(input_shape=base.output_shape[1:]))
        transfer.add(Dense(256,activation='relu'))
        transfer.add(Dense(7,activation='softmax'))
        print(base.output)
        print(squeeze(base.output,0))
        self.model = Model(inputs=base.input,output=transfer(base.output))

    def train(self,data,label,validation):
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.weight_name, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
        history = self.model.fit(data,label,epochs=self.epochs,batch_size=self.batch_size,validation_data=validation,shuffle=True,callbacks=[checkpoint])
        pd.DataFrame.from_dict(history.history).to_csv(self.history_name, float_format="%.5f", index=False)

    def test(self,data,label):
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        loss_and_metric = self.model.evaluate(data,label,batch_size=32) # 这里的batchsize其实无所谓= =
        print("{} on test set".format(self.name),loss_and_metric)

    def load(self):
        self.model.load_weights(self.weight_name)

    def print(self):
        self.model.summary()

flags = tf.app.flags
flags.DEFINE_string('VGG',"VGG16","decide the VGG version")
flags.DEFINE_integer('epochs',10,"epochs to train")
flags.DEFINE_integer('batch_size',32,"batch size to train")
flags.DEFINE_float('dropout',0,"dropout rate for full connected layer")
flags.DEFINE_boolean('is_train',True,"train or test")
configs = flags.FLAGS

def main(_):
    ## data
    train_x = np.load(TD)
    train_x = train_x.reshape((-1,48,48)) # 将每行的图片数据reshape成图片 (nx48x48)
    train_x = train_x[:,:,:,np.newaxis] # 将最后一维拓展成channel (nx48x48x1)
    train_y = np.load(TL)                   
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

    # 如果直接调用keras提供的vgg，需要将图片拓展到rb三个channel，这里直接复制三次
    if configs.VGG == vgg16pre:  
        train_x = np.array([train_x,train_x,train_x]).squeeze().transpose((1,2,3,0)) 
        validate_x = np.array([validate_x,validate_x,validate_x]).squeeze().transpose((1,2,3,0)) 
        test_x = np.array([test_x,test_x,test_x]).squeeze().transpose((1,2,3,0)) 
    elif configs.VGG == vgg19pre:
        train_x = np.array([train_x,train_x,train_x]).squeeze().transpose((1,2,3,0))
        validate_x = np.array([validate_x,validate_x,validate_x]).squeeze().transpose((1,2,3,0))
        test_x = np.array([test_x,test_x,test_x]).squeeze().transpose((1,2,3,0))
    ## model
    vgg = VGG(configs)
    vgg.print()
    if configs.is_train:
        vgg.train(train_x,train_y,(validate_x,validate_y))
        vgg.test(test_x,test_y)
    else:
        vgg.load()
        vgg.test(test_x,test_y)

if __name__ == "__main__":
    tf.app.run()
