from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization as BN, MaxPooling2D as MP
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt 

TD = "train_data.npy"
TL = "train_label.npy"
VD = "private_data.npy" # validation
VL = "private_label.npy"
PD = "public_data.npy"
PL = "public_label.npy"

class SuperEmotion(object):
    def __init__(self,configs):
        self.name = "super_emotion"
        self.weight_name = 'model/{}.hdf5'.format(self.name)
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
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
        sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.weight_name, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
        self.model.fit(data,label,epochs=self.epochs,batch_size=self.batch_size,validation_data=validation,shuffle=True,callbacks=[checkpoint])

    def test(self,data,label):
        loss_and_metric = self.model.evaluate(data,label,batch_size=32)
        print("{} on test set".format(self.name),loss_and_metric)

    def load(self):
        self.model.load_weights(self.weight_name)

    def print(self):
        print(self.model.summary)

flags = tf.app.flags
flags.DEFINE_integer('epochs',50,"epochs to train")
flags.DEFINE_integer('batch_size',128,"batch size to train")
flags.DEFINE_float('dropout',0.2,"dropout rate for full connected layer")
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

    ## model
    se = SuperEmotion(configs)
    se.print()
    if configs.is_train:
        se.train(train_x,train_y,(validate_x,validate_y))
        se.test(test_x,test_y)
    else:
        se.load()
        se.test(test_x,test_y)

if __name__ == "__main__":
    tf.app.run()

