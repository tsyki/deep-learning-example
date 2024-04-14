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

@np.vectorize
def sigmoid(x):
    #オーバーフロー対策
    #参考：https://www.kamishima.net/mlmpyja/lr/sigmoid.html
    sigmoid_range = 34.538776394910684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15
    
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策で入力値の最大値を引いておく
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    
# 2乗和誤差。値が小さいほど正解に近い
# y=ニューラルネットワークの出力、t=教師データの想定
def sum_squared_error(y,t):
    return 0.5 * np.sum((y-t) ** 2)

def show_sample_sum_squared_error():
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1 , 0.05 , 0.6 , 0.0 , 0.05 , 0.1 , 0.0 , 0.1 , 0.0 , 0.0]
    print('2乗和誤差:' + str( sum_squared_error(np.array(y), np.array(t)) ))

# 交差エントロピー誤差。値が小さいほど正解に近い
# y=ニューラルネットワークの出力、t=教師データの想定
def cross_entropy_error_simple(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))

def cross_entropy_error(y,t):
    delta = 1e-7
    #ndim=次元数=多次元配列の要素数。1要素の場合、[1,2,3]→[[1,2,3]]のように変換
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+delta)) / batch_size


def show_sample_cross_entropy_error_simple():
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1 , 0.05 , 0.6 , 0.0 , 0.05 , 0.1 , 0.0 , 0.1 , 0.0 , 0.0]
    print('交差エントロピー誤差:' + str( cross_entropy_error_simple(np.array(y), np.array(t)) ))

def show_sample_cross_entropy_error_batch():
    t = np.array([[0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])
    y = np.array([[0.1 , 0.05 , 0.6 , 0.0 , 0.05 , 0.1 , 0.0 , 0.1 , 0.0 , 0.0],
                 [0.9 , 0.1 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]
                 ])
    print('交差エントロピー誤差(2要素):' + str( cross_entropy_error(np.array(y), np.array(t)) ))
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1 , 0.05 , 0.6 , 0.0 , 0.05 , 0.1 , 0.0 , 0.1 , 0.0 , 0.0]
    print('交差エントロピー誤差(1要素):' + str( cross_entropy_error(np.array(y), np.array(t)) ))


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
    
#入力層2、第1層3、第2層2、出力層2のニューラルネットワーク
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    #print('x =' + str(x))
    #print('W1 =' + str(W1))
    #print(x.shape)
    #print(W1.shape)
    
    a1 = np.dot(x, W1) + b1
    print('a1:' + str(a1)) #1層の入力
    z1 = sigmoid(a1)
    print('z1:' + str(z1)) #1層の出力結果
    a2 = np.dot(z1, W2) + b2
    print('a2:' + str(a2)) #1層の入力
    z2 = sigmoid(a2)
    print('z2:' + str(z2)) #2層の出力結果
    a3 = np.dot(z2, W3) + b3
    #回帰問題は恒等関数、分類問題はソフトマックス関数を使う
    #y = identity_function(a3)
    y = softmax(a3)
    
    return y

def show_sample_network():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print('y:'+str(y))