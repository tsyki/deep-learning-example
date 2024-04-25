# -*- coding: utf-8 -*-

import numpy as np
import perceptron as per
import neural_network as nn
import minst_loader as miload
import predict_by_sample_data as predictsample
import gradient_simple_network as gsn
import gradient_two_layer_network as gtln

def main():
    print('AND(1,1):' + str(per.AND(1, 1)))
    print('AND(0,1):' + str(per.AND(0, 1)))
    
    print('AND(1,1):' + str( per.AND_simple(1, 1)))
    print('AND(0,1):' + str( per.AND_simple(0, 1)))
    
    x = np.array([0,1])
    w = np.array([0.5,0.5])
    b = -0.7
    print('perceptron sample:' +str(np.sum(w*x) + b))
    
    #3章 ニューラルネットワーク
    nn.show_step_function_sample()
    nn.show_sigmoid_function_sample()
    nn.show_relu_function_sample()
    nn.show_sample_network()
    miload.load_image()
    predictsample.print_predict_sample_first()
    predictsample.print_predict_sample()

    #4章 ニューラルネットワークの学習
    #2乗和誤差
    nn.show_sample_sum_squared_error()
    #交差エントロピー誤差
    nn.show_sample_cross_entropy_error_batch()
    nn.show_numerical_diff_sample()
    #勾配の計算
    nn.show_numerical_gradient_1d_sample()
    #勾配降下法での最小値の算出
    nn.show_gradient_descent_sample()
    #ニューラルネットワークの勾配の算出
    gsn.show_simple_network()
    gtln.show_two_layer_network()
    #ニューラルネットワークを使った学習
    gtln.training()
    

if __name__ == "__main__":
    main()