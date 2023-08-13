# -*- coding: utf-8 -*-

import numpy as np
import perceptron as per
import neural_network as nn
import minst_loader as miload

def main():
    print('AND(1,1):' + str(per.AND(1, 1)))
    print('AND(0,1):' + str(per.AND(0, 1)))
    
    print('AND(1,1):' + str( per.AND_simple(1, 1)))
    print('AND(0,1):' + str( per.AND_simple(0, 1)))
    
    x = np.array([0,1])
    w = np.array([0.5,0.5])
    b = -0.7
    print('perceptron sample:' +str(np.sum(w*x) + b))
    
    nn.show_step_function_sample()
    nn.show_sigmoid_function_sample()
    nn.show_relu_function_sample()
    nn.show_sample_network()
    miload.load_image()


if __name__ == "__main__":
    main()