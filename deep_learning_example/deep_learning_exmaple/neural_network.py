# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

# 配列未対応版
def step_function_simple(x):
    if x > 0:
        return 1
    else:
        return 0
    
def step_function(x):
    return np.array(x>0 , dtype=np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def show_step_function_sample():
    x = np.arange(-5.0,5.0,0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y軸の範囲
    plt.show()
    
def show_sigmoid_function_sample():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y軸の範囲
    plt.show()
    
def show_relu_function_sample():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 6.1) # y軸の範囲
    plt.show()