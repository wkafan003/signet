#%%
import numpy as np
import cv2
import os
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Input, Dropout, Flatten
from keras.models import Model
from keras.layers import Lambda
import subprocess
#%%

if os.path.exists('weights.zip') is False:
    print('Загрузка весов нейросети ...')
    import gdown
    gdown.download('https://drive.google.com/uc?id=1Udet-QX5YKEB01Fv-bHtGGJgPXDWXSgT')
else:
    print('Веса нейросети найдены.')


#%%

if os.path.exists('checkpoints') is False:
    print('Распаковка ...')
    subprocess.call('unzip weights.zip',shell=True)

#%%

from keras import backend as K


def euclidean_distance2(y):
    return K.sqrt(K.sum(K.square(y[0] - y[1]), axis=-1))




#%%

def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

#%%

def make_net():
    input = Input(shape=(155, 220, 3))

    conv_1 = Conv2D(filters=96, kernel_size=(11, 11))(input)
    batch_norm_1 = BatchNormalization()(conv_1)
    activation_1 = Activation('relu')(batch_norm_1)
    max_pool_1 = MaxPooling2D(pool_size=(3, 3))(activation_1)

    conv_2 = Conv2D(filters=256, kernel_size=(5, 5))(max_pool_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    activation_2 = Activation('relu')(batch_norm_2)
    max_pool_2 = MaxPooling2D(pool_size=(3, 3))(activation_1)

    dropout_1 = Dropout(rate=0.3)(max_pool_2)

    conv_3_a = Conv2D(filters=384, kernel_size=(3, 3))(dropout_1)
    activation_3_a = Activation('relu')(conv_3_a)
    conv_3_b = Conv2D(filters=256, kernel_size=(3, 3))(activation_3_a)
    activation_3_b = Activation('relu')(conv_3_b)
    max_pool_3 = MaxPooling2D(pool_size=(3, 3))(activation_3_b)

    # dropout_22 = Dropout(rate=0.3)(max_pool_3)
    # conv_4_a = Conv2D(filters=384, kernel_size=(3, 3))(dropout_22)
    # activation_4_a = Activation('relu')(conv_4_a)
    # conv_4_b = Conv2D(filters=512, kernel_size=(3, 3))(activation_4_a)
    # activation_4_b = Activation('relu')(conv_4_b)
    # max_pool_4 = MaxPooling2D(pool_size=(2, 2))(activation_4_b)

    dropout_2 = Dropout(rate=0.3)(max_pool_3)
    # dropout_2 = Dropout(rate=0.3)(max_pool_3)

    flat_1 = Flatten()(dropout_2)
    fc_1 = Dense(units=1024, activation='relu')(flat_1)
    dropout_3 = Dropout(rate=0.5)(fc_1)
    fc_2 = Dense(units=128, activation='relu')(dropout_3)



    input_a = Input(shape=(155, 220, 3))
    input_b = Input(shape=(155, 220, 3))

    base_net = Model(input, fc_2)
    processed_a = base_net(input_a)
    processed_b = base_net(input_b)

    distance = Lambda(euclidean_distance2)([processed_a, processed_b])
    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    return base_net,model

#%%
print('Загрузка весов')
base_net,model = make_net()
model.load_weights('checkpoints/best')

#%%
print('-1 для завершения работы')
while True:
    print('Введите путь до изображения 1 подписи')
    path = str(input())
    if path =='-1':
        break

    im_1 = cv2.imread(path)
    im_1 = cv2.resize(im_1,(220,155))
    im_1 =1- im_1/255.0
    im_1 = np.expand_dims(im_1,axis=0)

    print('Введите путь до изображения 2 подписи')
    path = str(input())
    if path =='-1':
        break
    im_2 = cv2.imread(path)
    im_2 = cv2.resize(im_2,(220,155))
    im_2 = 1-im_2/255.0
    im_2 = np.expand_dims(im_2,axis=0)
    y_pred = model.predict([im_1,im_2])
    print(f'Вероятность подделки {min(y_pred,1)*100}%')
    print(f'Вердикт нейросети - подпись {"настоящая" if y_pred<0.5 else "подделка"}')

