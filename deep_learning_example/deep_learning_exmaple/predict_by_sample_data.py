import sys, os
sys.path.append(os.pardir)
import numpy as np
import neural_network as nn
import minst_loader as miload
import pickle


def init_netowork():
    with open("dataset\sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = nn.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = nn.sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = nn.softmax(a3)
    return y

def print_predict_sample():
    x,t= miload.get_test_data(True, True, False)
    network = init_netowork()
    
    accuracy_cnt = 0
        
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) #最も大きい値を取得
        if p == t[i]:
            accuracy_cnt += 1
    
    print('Accuracy'+ str(float(accuracy_cnt) / len(x)))
    
#動作確認用に1つ目のテストデータと各層の重みとバイアスを表示する    
def print_predict_sample_first():
    x_all,t_all = miload.get_test_data(False, True, False)
    x=x_all[0]
    miload.load_image_by_array(x)
    network = init_netowork()
    
    print("入力", end=" ")
    print_shape(x, "x")
    
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    print("第1層の重み", end=" ")
    print_shape(W1, "w1")
    print("第1層のバイアス", end=" ")
    print_shape(b1, "b1")
    
    a1 = np.dot(x,W1) + b1
    
    print("第1層の活性化関数の引数", end=" ")
    print_shape(a1, "a1")

    z1 = nn.sigmoid(a1)

    print("第1層の出力", end=" ")
    print_shape(z1, "z1")

    print("第2層の重み", end=" ")
    print_shape(W2, "w2")
    print("第2層のバイアス", end=" ")
    print_shape(b2, "b2")
    
    a2 = np.dot(z1,W2) + b2

    print("第2層の活性化関数の引数", end=" ")
    print_shape(a2, "a2")

    z2 = nn.sigmoid(a2)

    print("第2層の出力", end=" ")
    print_shape(z2, "z2")

    print("第3層の重み", end=" ")
    print_shape(W3, "w3")
    print("第3層のバイアス", end=" ")
    print_shape(b3, "b3")

    a3 = np.dot(z2,W3) + b3

    print("第3層の活性化関数の引数", end=" ")
    print_shape(a3, "a3")
    
    y = nn.softmax(a3)

    print("出力", end=" ")
    print_shape(y, "y")
    
    
def print_shape(val, val_name):
    print(val_name + ".shape=" + str(val.shape) + " " + val_name + "=" + str(val))