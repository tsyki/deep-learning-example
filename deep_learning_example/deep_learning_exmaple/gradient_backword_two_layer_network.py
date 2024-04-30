# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)
import numpy as np
from neural_network import sigmoid,softmax,cross_entropy_error,numerical_gradient_2d
from minst_loader import load_mnist
from collections import OrderedDict

class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy
        
def show_multi_layer():
    #100円のリンゴを2個買った場合の各パラメータの影響度合の計算
    apple_unit_price = 100
    apple_num = 2
    tax = 1.1
    
    apple_layer = MultiLayer()
    tax_layer = MultiLayer()
    
    #順伝播
    apple_price = apple_layer.forward(apple_unit_price, apple_num)
    taxin_price = tax_layer.forward(apple_price, tax)
    
    print("税込額=", taxin_price)
    #逆伝播
    d_total_price = 1
    d_apple_price, d_tax = tax_layer.backward(d_total_price)
    d_apple_unit_price, d_apple_num = apple_layer.backward(d_apple_price)
    
    # 2.2 110 200
    print("リンゴの単価の微分=",d_apple_unit_price, " リンゴの個数の微分=", d_apple_num, " 税率の微分=" , d_tax)


class AddLayer:
    def __init__(self):
        pass
        
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy 

def show_add_and_multi_layer():
        #100円のリンゴを2個、150円のみかんを3個買った場合の各パラメータの影響度合の計算
        apple_unit_price = 100
        apple_num = 2
        orange_unit_price = 150
        orange_num = 3
        tax = 1.1
        
        apple_layer = MultiLayer()
        orange_layer = MultiLayer()
        apple_orange_add_layer = AddLayer()
        tax_layer = MultiLayer()
        
        #順伝播
        apple_price = apple_layer.forward(apple_unit_price, apple_num)
        orange_price = orange_layer.forward(orange_unit_price, orange_num)
        taxout_price = apple_orange_add_layer.forward(apple_price, orange_price)
        taxin_price = tax_layer.forward(taxout_price, tax)
        
        print("税込額", taxin_price)
        #逆伝播
        d_taxin_price = 1
        d_taxout_price, d_tax = tax_layer.backward(d_taxin_price)
        d_apple_price, d_orange_price = apple_orange_add_layer.backward(d_taxout_price)
        d_orange_unit_price, d_orange_num = orange_layer.backward(d_orange_price)
        d_apple_unit_price, d_apple_num = apple_layer.backward(d_apple_price)
        
        # 2.2 110 3.3 165 650
        print("リンゴの単価の微分=",d_apple_unit_price, " リンゴの個数の微分=", d_apple_num,
              "みかんの単価の微分=",d_orange_unit_price, " みかんの個数の微分=", d_orange_num, 
              " 税率の微分=" , d_tax)
       
class ReluLayer:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        #0以下は伝搬しないのでそのための配列を作成
        self.mask = (x <= 0)
        out = x.copy()
        #入力値の0以下の添え字部は値を0に
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        #入力時に0以下であった部分は逆伝播でも0なので0を設定
        dout[self.mask] = 0
        #0以下でない部分はそのまま返す
        dx = dout
        return dx 
    
class SigmoidLayer:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        #1 / (1 + np.exp(-x))
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        #y^2exp(-x) = y(1-y)
        dx = dout * (1.0 - self.out) * self.out
        return dx 
    
#ニューラルネットワークの行列の積(x*w+b)
class BatchAffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0)
        return dx 

#ソフトマックス関数+交差エントロピー誤差
class SoftmaxAndLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None #softmaxの出力
        self.t = None #教師データ(one-hotラベル想定)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx 


class BackwordTwoLayerNetwork:
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
        #NOTE 以下がレイヤ無し版と異なる
        #レイヤ作成
        self.layers = OrderedDict()
        self.layers['Affine1'] = BatchAffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = BatchAffineLayer(self.params['W2'], self.params['b2'])
        self.lastLayer=SoftmaxAndLossLayer()
    
    def predict(self,x):
        #NOTE 以下がレイヤ無し版と異なる
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        #W1, W2 = self.params['W1'], self.params['W2']
        #b1, b2 = self.params['b1'], self.params['b2']
        #
        #a1 = np.dot(x, W1) + b1
        #z1 = sigmoid(a1)
        #a2 = np.dot(z1, W2) + b2
        #y = softmax(a2)
        #return y

    def loss(self,x,t):
        y = self.predict(x)
        #NOTE 以下がレイヤ無し版と異なる
        loss = self.lastLayer.forward(y, t)
        #loss = cross_entropy_error(y, t)
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
    #NOTE 学習には利用しないが、レイヤを使った勾配計算の検算に使う
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
    
    #新規追加メソッド
    def backword_gradient(self,x,t):
        #forward
        #lastLayer.lossにlossの値が保持される
        self.loss(x, t)
        #backward
        dout=1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

def training():
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    #ハイパーパラメータ
    iters_num = 1000  # 学習を繰り返す回数
    train_size = x_train.shape[0] #トレーニングデータの画像ファイル総数
    batch_size = 100 #一回の学習でまとめて学習する画像ファイル数
    learning_rate = 0.1
    
    #学習経過の保持用
    train_loss_list= []
    
    #NOTE 逆伝播法のニューラルネットワークを利用
    network = BackwordTwoLayerNetwork(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        #トレーニング用の画像の個数から、バッチサイズだけランダムに選択(batch_sizeの要素数の配列ができる)
        batch_mask = np.random.choice(train_size,batch_size)
        #上記の添字に対応するデータを取得。(batch_size, input_size)の配列ができる
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        #NOTE 誤差逆伝播法で勾配を取得
        grads = network.backword_gradient(x_batch,t_batch)
        
        #パラメータを更新
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate * grads[key]
        #学習経過の記録
        loss = network.loss(x_batch, t_batch)
        accuracy = network.accuracy(x_batch, t_batch)
        if (i+1) %100 == 0:
            train_loss_list.append(str(loss) +" " + str(accuracy))
            print(i+1,"回目の学習終了 loss=" ,loss," accuracy=", accuracy )    

    print(train_loss_list)

#検算
def show_diff():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = BackwordTwoLayerNetwork(input_size=784, hidden_size=50, output_size=10)
    
    #0～3番目までの値を取得
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backword = network.backword_gradient(x_batch, t_batch)
    
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backword[key] - grad_numerical[key]))
        print(key,":", diff)

#計算過程の確認用の小さなニューラルネットワーク。入力層の要素2、隠れ層の要素3、出力層の要素2の想定
class SimpleBackwordTwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        #1階層目の重み(要素数=入力層×隠れ層)
        self.params['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        #1階層目のバイアス
        self.params['b1'] = np.array([0.1, 0.2, 0.3])
        #2階層目の重み(要素数=隠れ層×出力層の要素数)
        self.params['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        #2階層目のバイアス
        self.params['b2'] = np.array([0.1, 0.2])
        #NOTE 以下がレイヤ無し版と異なる
        #レイヤ作成
        self.layers = OrderedDict()
        self.layers['Affine1'] = BatchAffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = BatchAffineLayer(self.params['W2'], self.params['b2'])
        self.lastLayer=SoftmaxAndLossLayer()
    
    def predict(self,x):
        print("x=" , x)
        x = self.layers['Affine1'].forward(x)
        print("affine1.forward=" , x)
        x = self.layers['Relu1'].forward(x)
        print("Relu1.forward=" , x)
        x = self.layers['Affine2'].forward(x)
        print("affine2.forward=" , x)
        return x
        
        #レイヤ版
        #for layer in self.layers.values():
        #    x = layer.forward(x)
        #return x
        
        #非レイヤ版
        #W1, W2 = self.params['W1'], self.params['W2']
        #b1, b2 = self.params['b1'], self.params['b2']
        #
        #a1 = np.dot(x, W1) + b1
        #z1 = sigmoid(a1)
        #a2 = np.dot(z1, W2) + b2
        #y = softmax(a2)
        #return y

    def loss(self,x,t):
        y = self.predict(x)
        #NOTE 以下がレイヤ無し版と異なる
        loss = self.lastLayer.forward(y, t)
        print("SoftmaxAndLossLayer.forward(=loss)=" , loss)
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
    #NOTE 学習には利用しないが、レイヤを使った勾配計算の検算に使う
    def numerical_gradient(self,x,t):
        def loss_W(W):
            return self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        print('grad[W1]=', grads['W1'])
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        print('grad[b1]=', grads['b1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        print('grad[W2]=', grads['b2'])
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])
        print('grad[b2]=', grads['b2'])

        return grads
    
    #新規追加メソッド
    def backword_gradient(self,x,t):
        #forward
        #lastLayer.lossにlossの値が保持される
        self.loss(x, t)
        #backward
        dout=1
        dout = self.lastLayer.backward(dout)
        print('SoftmaxAndLossLayer.backword=',dout)
        
        dout = self.layers['Affine2'].backward(dout)
        print("affine2.backward=" , dout)

        dout = self.layers['Relu1'].backward(dout)
        print("Relu1.backward=" , dout)

        dout = self.layers['Affine1'].backward(dout)
        print("affine1.backward=" , dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        print('grad[W1]=', grads['W1'])
        grads['b1'] = self.layers['Affine1'].db
        print('grad[b1]=', grads['b1'])
        grads['W2'] = self.layers['Affine2'].dW
        print('grad[W2]=', grads['W2'])
        grads['b2'] = self.layers['Affine2'].db
        print('grad[b2]=', grads['b2'])
        
        return grads

def simple_training():
    #ハイパーパラメータ
    iters_num = 100  # 学習を繰り返す回数
    learning_rate = 0.1
    
    #NOTE 逆伝播法のニューラルネットワークを利用
    network = SimpleBackwordTwoLayerNetwork(input_size=2, hidden_size=3, output_size=2)

    for i in range(iters_num):
        x = np.array([[1,2], [2,1]]) #入力値
        t = np.array([[0,1], [1,0]]) #答えラベル
        
        #NOTE 誤差逆伝播法で勾配を取得
        grads = network.backword_gradient(x,t)
        
        #パラメータを更新
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate * grads[key]
        #学習経過の記録
        loss = network.loss(x, t)
        accuracy = network.accuracy(x, t)
        print(i+1,"回目の学習終了 loss=" ,loss," accuracy=", accuracy )    
