# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)
import numpy as np
from neural_network import softmax,cross_entropy_error,numerical_gradient_2d

class SimpleNetwork:
    def __init__(self):
        self.W = np.random.randn(2, 3) #重みの初期値をガウス分布で初期化
    
    def predict(self,x):
        return np.dot(x, self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

def show_simple_network():
    net = SimpleNetwork()
    x = np.array([0.6,0.9]) #入力
    t = np.array([0,0,1]) #正解ラベル

    print("network.W=" + str(net.W))
    #p = net.predict(x) #入力と重みからの計算結果(損失関数の中で計算しているものの確認用)
    #print("p=" + str(p))
    #print("最大値のインデックス=" + str(np.argmax(p))) #ネットワークの出した答え
    #print("loss=" + str(net.loss(x,t))) #答えと正解ラベルから出した損失関数の値
    # 引数Wはダミーで利用しない
    def f(W):
        return net.loss(x, t)
    dW = numerical_gradient_2d(f, net.W)
    print("勾配dw=" + str(dW))
