# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)
import numpy as np
from neural_network import softmax,sigmoid,cross_entropy_error,numerical_gradient_2d
from minst_loader import load_mnist

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        #1階層目の重み(要素数=入力層×隠れ層)をガウス分布で初期化
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        #1階層目のバイアス(初期値は全て0)
        self.params['b1'] = np.zeros(hidden_size)
        #2階層目の重み(要素数=隠れ層×出力層の要素数)をガウス分布で初期化
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        #2階層目のバイアス(初期値は全て0)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self,x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self,x,t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss
    
    #認識精度
    def accuracy(self,x,t):
        y = self.predict(x)
        # 出力層の最大値=推測した値
        y = np.argmax(y, axis=1)
        # 答え
        t = np.argmax(t, axis=1)
        # 答えと一致した個数 / 入力要素数
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    #各重み、バイアスの勾配を計算
    def numerical_gradient(self,x,t):
        def loss_W(W):
            return self.loss(x,t)
        grads = {}
        print("W1の勾配計算開始")
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        print("b1の勾配計算開始")
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        print("W2の勾配計算開始")
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        print("b2の勾配計算開始")
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])

        return grads
        

def show_two_layer_network():
    print("2階層のニューラルネットワークの勾配計算開始")
    net = TwoLayerNetwork(input_size = 784,hidden_size = 100,output_size = 10)
    x = np.random.rand(2,784) #ダミーの入力(2個)
    t = np.random.rand(2,10) #ダミーの正解(2個)

    grads = net.numerical_gradient(x,t)
    print("勾配=" + str(grads))

def training():
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    #ハイパーパラメータ
    iters_num = 10000  # 学習を繰り返す回数
    train_size = x_train.shape[0] #トレーニングデータの画像ファイル総数
    batch_size = 10 #一回の学習でまとめて学習する画像ファイル数
    learning_rate = 0.1
    
    #学習経過の保持用
    train_loss_list= []
    
    network = TwoLayerNetwork(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        #トレーニング用の画像の個数から、バッチサイズだけランダムに選択(batch_sizeの要素数の配列ができる)
        batch_mask = np.random.choice(train_size,batch_size)
        #上記の添字に対応するデータを取得。(batch_size, input_size)の配列ができる
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        #各画像データの損失関数を計算し、勾配を取得
        grads = network.numerical_gradient(x_batch,t_batch)
        
        #パラメータを更新
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate * grads[key]
        #学習経過の記録
        loss = network.loss(x_batch, t_batch)
        accuracy = network.accuracy(x_batch, t_batch)
        train_loss_list.append(str(loss) +" " + str(accuracy))
        print(i+1,"回目の学習終了 loss=" ,loss," accuracy=", accuracy )    

    print(train_loss_list)    