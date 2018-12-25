# -*- coding: utf-8 -*-
# 評価, plt 表示、　csv -out
#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
from util_df import *
import time
import pickle

#

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
#quit()

# load model
#network = SimpleNet(input_size=1 , hidden_size=10, output_size=1 )
network = SimpleNet(input_size=5 , hidden_size=10, output_size=1 )
network.load_params("params.pkl" )
#
#pred
print( y_test[: 10 ] *num_max_y )
y_val = network.predict( x_test )
y_val = y_val * num_max_y
print(y_val[: 10] )
#    quit()

#y_train = y_train * num_max_y
#y_val   = y_val * num_max_y    
print ('time : ', time.time() - global_start_time)
#quit()

#print(y_val[:10] )
#print(x_test_dt[:10] )
#quit()
#print(len( y_test ))
#df = DataFrame(y_test , y_val  )
df = DataFrame(  y_test * num_max_y )
df.columns=["price"]
#df["c1"] =0
df["pred"] =y_val
df["diff"] = df["price"] - df["pred"]
#print( df.head() )
#print( type(df)  )
#print( df[: 10] )
#print(len( y_val ))
print(df["diff"].max() )
print(df["diff"].min() )
print(df["diff"].mean() )
df.to_csv('pred_2_y.csv', index=False)
#quit()
#plt
#plt.plot(x_test, y_test * num_max_y, "o",label = "y_test" )
a1=np.arange(len(y_val) )
plt.plot(a1 , y_test *num_max_y , label = "y_test")
plt.plot(a1 , y_val , label = "predict")
#    plt.scatter(x_test , y_val )
plt.legend()
plt.grid(True)
plt.title("price pred")
plt.xlabel("x")
plt.ylabel("price")
plt.show()
