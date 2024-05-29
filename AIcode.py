from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/archive')
os.listdir()

#########################################################################################################################################################

import pandas as pd
import cv2
from tensorflow import keras
from glob import glob
import os
import numpy as np

birds_latin_names=pd.read_csv('birds latin names.csv')
idx_to_class=birds_latin_names.iloc[:200,1].values

class_count=len(idx_to_class)

tmp=[(idx_to_class[i],i) for i in range(class_count)]
class_to_idx=dict(tmp)

def load_data_set(path:str,data_set:str,class_list:list):
    x=[]
    y=[]
    for class_name in class_list:
        path_list=glob(os.path.join(path,data_set,class_name,'*'))
        for tmp in path_list:
            img=cv2.imread(tmp)
            img=cv2.resize(img,(112,112))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            x.append(img)
            y.append(class_to_idx[class_name])
    y=np.array(y)
    y=keras.utils.to_categorical(y,class_count)
    return np.array(x),y

x_train,y_train=load_data_set('','train',idx_to_class)
x_test,y_test=load_data_set('','test',idx_to_class)
x_valid,y_valid=load_data_set('','valid',idx_to_class)

print(x_train.shape)
print(y_train.shape)

np.save('x_train200',x_train)
np.save('y_train200',y_train)

np.save('x_valid200',x_valid)
np.save('y_valid200',y_valid)

np.save('x_test200',x_test)
np.save('y_test200',y_test)

#########################################################################################################################################################

from tensorflow import keras
import numpy as np
from keras.models import *
from keras.layers import *

x_train,y_train=np.load('x_train200.npy'),np.load('y_train200.npy')
x_valid,y_valid=np.load('x_valid200.npy'),np.load('y_valid200.npy')
x_test,y_test=np.load('x_test200.npy'),np.load('y_test200.npy')

def VGG16(data, classes):

    inputs = Input(shape=data)

    base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)


    base_model = keras.Model(inputs, base_model.layers[-1].output)

    base_model.trainable = False
    x = base_model(inputs)

    fc = Flatten()(x)
    fc = Dense(2048, activation='relu')(fc)
    fc = Dense(2048, activation='relu')(fc)

    output = Dense(classes, activation='softmax')(fc)
    model = keras.Model(inputs, output)

    return model

model = VGG16((112,112,3),200)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
batch_size = 512
epochs = 100
my_callbacks=[
              keras.callbacks.EarlyStopping(
                  patience=10,
                  monitor='val_accuracy',
                  restore_best_weights=True
              )
]

from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(
    rotation_range=0,
    horizontal_flip=True,
    width_shift_range=0.2
)

result=model.fit(
    datagen.flow(x_train,y_train,batch_size=batch_size),
    validation_data=(x_valid,y_valid),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=my_callbacks,
    shuffle=True
)

model.evaluate(x_test,y_test)

model.save('model17')

#########################################################################################################################################################

import tensorflow as tf
from tensorflow import keras

os.chdir('/content/drive/MyDrive')

model=keras.models.load_model('model10')

import pandas as pd
os.chdir('/content/drive/MyDrive/Bird_Data/archive')
birds_latin_names=pd.read_csv('birds latin names.csv')
idx_to_class=birds_latin_names.iloc[:200,1].values

class_count=len(idx_to_class)

tmp=[(idx_to_class[i],i) for i in range(class_count)]
class_to_idx=dict(tmp)

import numpy as np
x_test,y_test=np.load('x_test200.npy'),np.load('y_test200.npy')

import matplotlib.pyplot as plt
def show_img(img_to_show):
    plt.axis('off')
    if len(img_to_show.shape)==3:plt.imshow(img_to_show)
    else: plt.imshow(img_to_show,cmap='gray')
    plt.show()

print(model.evaluate(x_test,y_test))

from sklearn.metrics import precision_recall_fscore_support
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
y_test_class=np.argmax(y_test,axis=1)
print('(precision, recall, fscore)')
print(precision_recall_fscore_support(y_test_class,y_pred,average='macro'))

import hashlib

data_num=x_test.shape[0]
print('total',data_num)

while True:
    tmp=input('enter a number : ')
    try:tmp=int(tmp)
    except ValueError:
        sha3_256=hashlib.sha3_256()
        sha3_256.update(tmp.encode())
        tmp=int.from_bytes(sha3_256.digest(),byteorder='big')
    n=tmp%data_num
    x=x_test[n,:,:]
    x=np.expand_dims(x,0)
    print(x.shape)
    y_pred=model.predict(x)

    pre_ans=np.argmax(y_pred)
    ans=np.argmax(y_test[n])
    print('idx :',n)
    print('AI  :',idx_to_class[pre_ans],'| class_idx :',pre_ans)
    print('ans :',idx_to_class[ans],'| class idx :',ans)
    show_img(x_test[n])
