# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)
import numpy as np
from neural_network import softmax,sigmoid,cross_entropy_error,numerical_gradient_2d
from minst_loader import load_mnist

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
       