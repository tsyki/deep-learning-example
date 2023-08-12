# -*- coding: utf-8 -*-

import numpy as np
import perceptron as per
import neural_network as nn

def main():
    print( per.AND(1, 1))
    print( per.AND(0, 1))
    
    print( per.AND_simple(1, 1))
    print( per.AND_simple(0, 1))
    
    x = np.array([0,1])
    w = np.array([0.5,0.5])
    b = -0.7
    print( np.sum(w*x) + b)
    
    nn.show_step_function_sample()
    nn.show_sigmoid_function_sample()
    nn.show_relu_function_sample()


if __name__ == "__main__":
    main()