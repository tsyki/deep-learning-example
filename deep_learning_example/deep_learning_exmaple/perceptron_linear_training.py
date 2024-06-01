# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)
import numpy as np
from neural_network import step_function, numerical_gradient_2d

#隠れ層なしの単純パーセプトロン
class SimpleLinearPerceptron:
    def __init__(self, input_size, output_size, weight_init_std = 0.01):
        self.params = {}
        #1階層目の重み(要素数=入力層×出力層)をガウス分布で初期化
        #self.params['W1'] = weight_init_std * np.random.randn(input_size,output_size)
        #1階層目のバイアス(初期値は全て0)
        #self.params['b1'] = np.zeros(output_size)
        #1階層目の重み(要素数=入力層×出力層)
        self.params['W1'] = np.array([0.8, 0.5])
        #1階層目のバイアス
        self.params['b1'] = np.array([-0.7])
    
    def predict(self,x):
        W1 = self.params['W1']
        b1 = self.params['b1']
        
        a1 = np.dot(x, W1) + b1
        #z1 = sigmoid(a1)
        #a2 = np.dot(z1, W2) + b2
        #y = softmax(a1)
        #yの値は変換しない(=活性化関数が恒等関数である想定)
        y = a1
        return y


    def loss(self,x,t):
        y = self.predict(x)
        #NOTE クロスエントロピー誤差は使えないので適当な損失関数を作る
        #loss = cross_entropy_error(y, t)
        loss = 0
        #適当に全ての要素に対する損失関数の合計を返すことにする
        #この実装だと入力値が巨大な場合にその誤差が大きく出てしまうが考慮しない
        for i in range(y.size):
            loss += self.calc_loss(y[i], t[i])
        return loss
    
    def calc_loss(self,y,t):
        #値の差異が小さければ小さいほどよい
        return abs(y - t)
    
    def is_correct(self,y,t):
        #ほとんど同じ値になったら正解とみなす
        threshold = 0.1
        if ( abs(y - t) < threshold ):
            return True
        else:
            return False
    
    #認識精度
    def accuracy(self,x,t):
        y = self.predict(x)
        # 出力層の最大値=推測した値
        #y = np.argmax(y, axis=1)
        # 答え
        #t = np.argmax(t, axis=1)
        # 答えと一致した個数 / 入力要素数
        #accuracy = np.sum(y == t) / float(x.shape[0])
        
        correct_num = 0
        for i in range(y.size):
            if self.is_correct(y[i], t[i]):
                correct_num = correct_num + 1
        
        accuracy = correct_num / float(x.shape[0])

        return accuracy
    
    #各重み、バイアスの勾配を計算
    #NOTE 学習には利用しないが、レイヤを使った勾配計算の検算に使う
    def numerical_gradient(self,x,t):
        def loss_W(W):
            return self.loss(x,t)
        grads = {}
        print("W1の勾配計算開始")
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        print("W1の勾配=", grads['W1'])
        print("b1の勾配計算開始")
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        print("b1の勾配=", grads['b1'])

        return grads

def simple_training_linear():
    #ハイパーパラメータ
    iters_num = 100  # 学習を繰り返す回数
    learning_rate = 0.01
    
    network = SimpleLinearPerceptron(input_size=2, output_size=1)

    for i in range(iters_num):
        
        x = np.array([[1.0,1.0], [1.0,0.0], [0.0,1.0] , [0.0,0.0] ]) #入力値
        t = np.array([[4],[3],[2],[1]]) #答えラベル 2x1 + x2 + 1の想定
        
        #NOTE 勾配法で勾配を取得
        grads = network.numerical_gradient(x,t)
        
        #パラメータを更新
        for key in ('W1','b1'):
            network.params[key] -= learning_rate * grads[key]
        #学習経過の記録
        loss = network.loss(x, t)
        accuracy = network.accuracy(x, t)
        print(i+1,"回目の学習終了 loss=" ,loss," accuracy=", accuracy, " W1=",network.params['W1'], " b1=", network.params['b1'] )    
