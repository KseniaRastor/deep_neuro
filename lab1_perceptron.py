# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# загружаем и подготавляваем данные
df = pd.read_csv('data_err.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# создаем матрицу весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

#Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
   
# функция прямого прохода (предсказания) 
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict


pr_r = predict(3)


# обучение
# у перцептрона Розенблатта обучаются только веса выходного слоя 
# как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки
n_iter=5
eta = 0.01
myLoop = True
my_itrrations = 0

#count = 0
arr_err = []
#repeat = 1
#repeat1 = 1
#saWout_copy = np.zeros((1+hiddenSizes,outputSize))

arrWouts = []   
#array = [0] *5

#array.append(2)
#array.append(2)

#if (not(array.count(-2) or array.count(2))):
#    print("Нули")
#else:
#    print("Ошибки есть")


 
while (myLoop == True):  
    arr_err.clear()
    my_itrrations += 1
 
    
    if (arrWouts.count(Wout)):
        checkW = True              #вектор есть -> зацикливание    
    else:
        arrWouts.append(Wout)
        checkW = False
    
    
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi)                                          
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)       
        Wout[0] += eta * (target - pr)
        
        
        arr_err.append(target - pr) 
        
    if (not(arr_err.count(-2) or arr_err.count(2))):  
        myLoop = False
        print("Ошибок нет")
    else:
        if (checkW == True):
            myLoop = False
            print("Зацикливание")
        else:
            myLoop = True
            
            
    #if ((all(s != aarr_err[0] for s in arr_err)) and (checkW == False)) :
        
    #if ((2 or -2 in arr_err) and (checkW == False)):                            #пока есть ошибки
    #    myLoop= True                                                           #и вектор весов не повторяется
    #else:
    #    myLoop = False
    
    #if (2 or -2 in arr_err):

        
    # if ((arr_err.count(-2) or arr_err.count(2)) and (checkW == False)):
    #     myLoop= True
    # else:
    #     myLoop = False
        
    #     if (checkW == True):
    #         print("Зацикливание")
            
    #     if (arr_err.count(-2) or arr_err.count(2)):    
    #         print("Ошибки есть")
    #     else:
    #         print("Ошибок нет")
            
        
        
            
    
       

# посчитаем сколько ошибок делаем на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)
otvet = pr-y.reshape(-1, 1)
sum(pr-y.reshape(-1, 1))/2

# далее оформляем все это в виде отдельного класса neural.py
