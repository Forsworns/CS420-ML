from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization as BN, MaxPooling2D as MP
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
from configs import *
import pandas as pd
import matplotlib.pyplot as plt 


class Emotion(object):
    def __init__(self,configs):
        self.name = "emotion"
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.weight_name = 'model/{}-{}-{}-{}.hdf5'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.history_name = 'history/{}-{}-{}-{}.csv'.format(self.name,self.epochs,self.batch_size,self.dropout)
        print(self.epochs)
        self.build_model()

    def build_model(self):
        self.model=Sequential()        
        self.model.add(Conv2D(32,(1,1),strides=1,padding='same',input_shape=(48,48,1)))        
        self.model.add(Activation('relu'))        
        self.model.add(Conv2D(32,(5,5),padding='same'))        
        self.model.add(Activation('relu'))        
        self.model.add(MP(pool_size=(2,2)))                
        self.model.add(Conv2D(32,(3,3),padding='same'))        
        self.model.add(Activation('relu'))        
        self.model.add(MP(pool_size=(2,2)))         
        self.model.add(Conv2D(64,(5,5),padding='same'))        
        self.model.add(Activation('relu'))        
        self.model.add(MP(pool_size=(2,2)))                
        self.model.add(Flatten())        
        self.model.add(Dense(2048))        
        self.model.add(Activation('relu'))        
        self.model.add(Dropout(0.5))        
        self.model.add(Dense(1024))        
        self.model.add(Activation('relu'))        
        self.model.add(Dropout(0.5))        
        self.model.add(Dense(7))        
        self.model.add(Activation('softmax'))

    def train(self,data,label,validation):
        sgd=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
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
        plot_model(self.model, to_file="{}.png".format(self.name), show_shapes=True)
        plt.show()

flags = tf.app.flags
flags.DEFINE_integer('epochs',50,"epochs to train")
flags.DEFINE_integer('batch_size',128,"batch size to train")
flags.DEFINE_float('dropout',0,"dropout rate for full connected layer")
flags.DEFINE_boolean('is_train',True,"train or test")
flags.DEFINE_boolean('fool',True,"fool or test")
flags.DEFINE_boolean('augment',False,"augment or test")
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
    if configs.augment:
        # 翻转
        augment = np.fliplr(train_x)
        train_x = np.vstack((train_x,augment))
        train_y = np.concatenate((train_y,train_y))
        augment = np.flipud(train_x)
        train_x = np.vstack((train_x,augment))
        train_y = np.concatenate((train_y,train_y))
        # 改变灰度
        augment = np.array(train_x*0.6,dtype='uint8')
        train_x = np.vstack((train_x,augment))
        train_y = np.concatenate((train_y,train_y))
        augment = np.clip(np.array(train_x*1.2,dtype='uint8'),0,255)
        train_x = np.vstack((train_x,augment))
        train_y = np.concatenate((train_y,train_y))
        # 加噪声
        augment = np.zeros_like(train_x)
        for n in range(train_x.shape[0]):
            noise = np.random.randint(-5, 5, size = (48, 48))
            augment[n,...] = np.clip(train_x[n,...].squeeze()+noise,0,255)
        train_x = np.vstack((train_x,augment))
        train_y = np.concatenate((train_y,train_y))
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

    ## model
    e = Emotion(configs)
    e.print()
    '''
    if configs.is_train:
        e.train(train_x,train_y,(validate_x,validate_y))
        e.test(test_x,test_y)
    else:
        e.load()
        e.test(test_x,test_y)
    '''

if __name__ == "__main__":
    tf.app.run()

