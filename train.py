# -*- coding: utf-8 -*-
# max_x= max_y の場合。
# 2018/12/23 15:46: 入力層の、形状変更
# DL, pred_2-ML の, DL 版(体重の予測)
# train/学習処理。結果ファイル保存。
# TwoLayerNet を参考に、３層ネットワーク利用
#  学習　>パラメータ保存

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
from util_df import *
import time

#
# 学習データ
# 学習データ
global_start_time = time.time()
wdata = pd.read_csv("data.csv" )
wdata.columns =["no", "price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ,"madori" ,"houi" ,"kouzou" ]
#print(wdata.head() )
#quit()

# conv=> num
sub_data = wdata[[ "no","price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ] ]
sub_data = sub_data.assign(price=pd.to_numeric( sub_data.price))
print( sub_data.head() )
print(sub_data["price"][: 10])

# 説明変数に "price" 以外を利用
X = sub_data.drop("price", axis=1)
X = X.drop("no", axis=1)

#num_max_x= 10
num_max_x= 1000
X = (X / num_max_x )
print(X.head() )
print(X.shape )
#print( type( X) )
#print(X[: 10 ] )

# 目的変数
num_max_y= num_max_x
Y = sub_data["price"]
Y = Y / num_max_y
print(Y.max() )
print(Y.min() )
#quit()

# 学習データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
#x_train_sub =x_train
#x_test_sub  =x_test
#x_train = x_train["x_dat"]
#x_test = x_test["x_dat"]
#print(type(x_train) )
#quit()
x_train =np.array(x_train, dtype = np.float32).reshape(len(x_train), 5)
y_train =np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
x_test  =np.array(x_test, dtype  = np.float32).reshape(len(x_test), 5 )
y_test =np.array(y_test, dtype   = np.float32).reshape(len(y_test), 1)
#
#x_train =np.array(x_train, dtype = np.float64 ).reshape(len(x_train), 5)
#y_train =np.array(y_train, dtype = np.float64).reshape(len(y_train), 1)
#x_test  =np.array(x_test, dtype  = np.float64).reshape(len(x_test), 5 )
#y_test =np.array(y_test, dtype   = np.float64).reshape(len(y_test), 1)

print( x_train.shape , y_train.shape  )
print( x_test.shape  , y_test.shape  )
#print(x_train[: 10])
#print(type(x_train ))
#quit()
#
network = SimpleNet(input_size=5 , hidden_size=10, output_size=1 )

#iters_num = 30000  # 繰り返しの回数を適宜設定する    
iters_num = 10000  # 繰り返しの回数を適宜設定する    

train_size = x_train.shape[0]
print( train_size )
#quit()

#
global_start_time = time.time()

#batch_size = 100
#batch_size = 32
batch_size = 16
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
#
#iter_per_epoch =200
iter_per_epoch = 500

#print(iter_per_epoch)
#quit()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask )
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]
    #quit()s
    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        print ("i=" +str(i) + ', time : '+ str( time.time() - global_start_time) + " , loss=" +str(loss))

#print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
print ('time : ', time.time() - global_start_time)
#
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")
#quit()

#pred
y_test_div=y_test[: 10] * num_max_y
#print( y_test_div )
print( y_test_div )
y_val = network.predict(x_test[: 10])
y_val = y_val * num_max_y
print( y_val )

