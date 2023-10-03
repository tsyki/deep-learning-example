# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
print('x_train.shape:' + str(x_train.shape))
print('t_train.shape:' + str(t_train.shape))
print('x_test.shape:' + str(x_test.shape))
print('t_test.shape:' + str(t_test.shape))

def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def load_image_by_array(img):
    img = img.reshape(28,28) #形状を元の画像サイズに変形
    image_show(img)
    

def load_image():
    img = x_train[0]
    label = t_train[0]
    print('t_train[0](label):' + str(label))
    
    print('x_train[0](img).shape:' + str(img.shape))
    img = img.reshape(28,28) #形状を元の画像サイズに変形
    print('x_train[0](img).reshape.shape:' + str(img.shape))
    
    image_show(img)

