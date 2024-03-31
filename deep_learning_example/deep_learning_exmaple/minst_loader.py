# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

# XXX 引数名と仮引数名が重複する場合ってどうするもの？
def get_training_data(normalize_,flatten_, one_hot_label_):
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=normalize_,flatten =flatten_,one_hot_label=one_hot_label_)
    return x_train, t_train

def get_test_data(normalize_,flatten_, one_hot_label_):
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=normalize_,flatten =flatten_,one_hot_label=one_hot_label_)
    return x_test, t_test

def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def load_image_by_array(img):
    img = img.reshape(28,28) #形状を元の画像サイズに変形
    image_show(img)
    

def load_image():
    x_train,t_train = get_training_data(False,True,False)
    img = x_train[0]
    label = t_train[0]
    print('t_train[0](label):' + str(label))
    
    print('x_train[0](img).shape:' + str(img.shape))
    img = img.reshape(28,28) #形状を元の画像サイズに変形
    print('x_train[0](img).reshape.shape:' + str(img.shape))
    
    image_show(img)

