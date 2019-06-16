from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
# from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from configs import *
import cv2

class Inception_v4(object):
    def __init__(self,configs):
        self.name = "inception_v4"
        self.epochs = configs.epochs
        self.dropout = configs.dropout
        self.batch_size = configs.batch_size
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.weight_name = 'model/{}-{}-{}-{}.hdf5'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.history_name = 'history/{}-{}-{}-{}.csv'.format(self.name,self.epochs,self.batch_size,self.dropout)
        self.build()

    def conv_block(self, x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1
        x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x


    def inception_stem(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
        x = self.conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
        x = self.conv_block(x, 32, 3, 3, border_mode='valid')
        x = self.conv_block(x, 64, 3, 3)

        x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
        x2 = self.conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

        x = concatenate([x1, x2], axis=channel_axis)

        x1 = self.conv_block(x, 64, 1, 1)
        x1 = self.conv_block(x1, 96, 3, 3, border_mode='valid')

        x2 = self.conv_block(x, 64, 1, 1)
        x2 = self.conv_block(x2, 64, 1, 7)
        x2 = self.conv_block(x2, 64, 7, 1)
        x2 = self.conv_block(x2, 96, 3, 3, border_mode='valid')

        x = concatenate([x1, x2], axis=channel_axis)

        x1 = self.conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
        x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

        x = concatenate([x1, x2], axis=channel_axis)
        return x


    def inception_A(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        a1 = self.conv_block(input, 96, 1, 1)

        a2 = self.conv_block(input, 64, 1, 1)
        a2 = self.conv_block(a2, 96, 3, 3)

        a3 = self.conv_block(input, 64, 1, 1)
        a3 = self.conv_block(a3, 96, 3, 3)
        a3 = self.conv_block(a3, 96, 3, 3)

        a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
        a4 = self.conv_block(a4, 96, 1, 1)

        m = concatenate([a1, a2, a3, a4], axis=channel_axis)
        return m


    def inception_B(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        b1 = self.conv_block(input, 384, 1, 1)

        b2 = self.conv_block(input, 192, 1, 1)
        b2 = self.conv_block(b2, 224, 1, 7)
        b2 = self.conv_block(b2, 256, 7, 1)

        b3 = self.conv_block(input, 192, 1, 1)
        b3 = self.conv_block(b3, 192, 7, 1)
        b3 = self.conv_block(b3, 224, 1, 7)
        b3 = self.conv_block(b3, 224, 7, 1)
        b3 = self.conv_block(b3, 256, 1, 7)

        b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
        b4 = self.conv_block(b4, 128, 1, 1)

        m = concatenate([b1, b2, b3, b4], axis=channel_axis)
        return m


    def inception_C(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        c1 = self.conv_block(input, 256, 1, 1)

        c2 = self.conv_block(input, 384, 1, 1)
        c2_1 = self.conv_block(c2, 256, 1, 3)
        c2_2 = self.conv_block(c2, 256, 3, 1)
        c2 = concatenate([c2_1, c2_2], axis=channel_axis)

        c3 = self.conv_block(input, 384, 1, 1)
        c3 = self.conv_block(c3, 448, 3, 1)
        c3 = self.conv_block(c3, 512, 1, 3)
        c3_1 = self.conv_block(c3, 256, 1, 3)
        c3_2 = self.conv_block(c3, 256, 3, 1)
        c3 = concatenate([c3_1, c3_2], axis=channel_axis)

        c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
        c4 = self.conv_block(c4, 256, 1, 1)

        m = concatenate([c1, c2, c3, c4], axis=channel_axis)
        return m


    def reduction_A(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        r1 = self.conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

        r2 = self.conv_block(input, 192, 1, 1)
        r2 = self.conv_block(r2, 224, 3, 3)
        r2 = self.conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

        r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

        m = concatenate([r1, r2, r3], axis=channel_axis)
        return m


    def reduction_B(self,input):
        if K.image_dim_ordering() == "th":
            channel_axis = 1
        else:
            channel_axis = -1

        r1 = self.conv_block(input, 192, 1, 1)
        r1 = self.conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

        r2 = self.conv_block(input, 256, 1, 1)
        r2 = self.conv_block(r2, 256, 1, 7)
        r2 = self.conv_block(r2, 320, 7, 1)
        r2 = self.conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

        r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

        m = concatenate([r1, r2, r3], axis=channel_axis)
        return m


    def build(self):
        init = Input((299, 299, 3))
        x = self.inception_stem(init)
        # 4 x Inception A
        for _ in range(4):
            x = self.inception_A(x)
        # Reduction A
        x = self.reduction_A(x)
        # 7 x Inception B
        for _ in range(7):
            x = self.inception_B(x)
        # Reduction B
        x = self.reduction_B(x)
        # 3 x Inception C
        for _ in range(3):
            x = self.inception_C(x)
        # Average Pooling
        x = AveragePooling2D((8, 8))(x)
        # Dropout
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        # Output
        out = Dense(7, activation='softmax')(x)
        self.model = Model(init, out, name=self.name)

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
flags.DEFINE_integer('epochs',10,"epochs to train")
flags.DEFINE_integer('batch_size',32,"batch size to train")
flags.DEFINE_float('dropout',0.8,"dropout rate for full connected layer")
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

    # 需要将图片拓展到rb三个channel，这里直接复制三次
    train_x = np.array([train_x,train_x,train_x]).squeeze().transpose((1,2,3,0)) 
    validate_x = np.array([validate_x,validate_x,validate_x]).squeeze().transpose((1,2,3,0)) 
    test_x = np.array([test_x,test_x,test_x]).squeeze().transpose((1,2,3,0)) 
    
    # reshape to 299x299
    train_x_l = np.zeros((train_x.shape[0],299,299,3))
    validate_x_l = np.zeros((validate_x.shape[0],299,299,3))
    test_x_l = np.zeros((test_x.shape[0],299,299,3))
    for n in range(train_x.shape[0]):
        train_x_l[n,...] = cv2.resize(np.array(train_x[n,...].squeeze(),dtype='uint8'), (299, 299), interpolation=cv2.INTER_CUBIC)
    for n in range(validate_x.shape[0]):    
        validate_x_l[n,...] = cv2.resize(np.array(validate_x[n,...].squeeze(),dtype='uint8'), (299, 299), interpolation=cv2.INTER_CUBIC)
    for n in range(test_x.shape[0]):    
        test_x_l[n,...] = cv2.resize(np.array(test_x[n,...].squeeze(),dtype='uint8'), (299, 299), interpolation=cv2.INTER_CUBIC)
    
    ## model
    inception_v4 = Inception_v4(configs)
    inception_v4.print()
    if configs.is_train:
        inception_v4.train(train_x_l,train_y,(validate_x_l,validate_y))
        inception_v4.test(test_x_l,test_y)
    else:
        inception_v4.load()
        inception_v4.test(test_x_l,test_y)

if __name__ == "__main__":
    tf.app.run()

