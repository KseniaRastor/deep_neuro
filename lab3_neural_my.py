import numpy as np
import torch
import torch.nn as nn
import pandas as pd

x = torch.randn(10, 3)
y = torch.randn(10, 2)

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self


class MLP:
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes = 5):

        # инициализируем нейронную сеть 
        # веса инициализируем случайными числами, но теперь будем хранить их списком
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),  # веса скрытого слоя
            np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))  # веса выходного слоя
        ]
        self.learning_rate = learning_rate
        self.layers = None

    # сигмоида
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # нам понадобится производная от сигмоиды при вычислении градиента
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
     
    # прямой проход 
    def feed_forward(self, x):
        input_ = x # входные сигналы
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0])) # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))# выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)
        
        self.layers = [input_, hidden_, output_]
        return self.layers[-1]
    
   
    # backprop собственной персоной
    # на вход принимает скорость обучения, реальные ответы, предсказанные сетью ответы и выходы всех слоев после прямого прохода
    def backward(self, target):
    
        # считаем производную ошибки сети
        err = (target - self.layers[-1])
    
        # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
        # для этого используем chain rule
        # цикл перебирает слои от последнего к первому
        for i in range(len(self.layers)-1, 0, -1):
            # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя
            
            # ошибка слоя * производную функции активации
            err_delta = err * self.derivative_sigmoid(self.layers[i])       
            
            # пробрасываем ошибку на предыдущий слой
            err = np.dot(err_delta, self.weights[i - 1].T)
            
            # ошибка слоя * производную функции активации * на входные сигналы слоя
            dw = np.dot(self.layers[i - 1].T, err_delta)
            
            # обновляем веса слоя
            self.weights[i - 1] += self.learning_rate * dw
            
            
    
    # функция обучения чередует прямой и обратный проход
    def train(self, x_values, target):
        self.feed_forward(x_values)
        self.backward(target)
    
    # функция предсказания возвращает только выход последнего слоя
    def predict(self, x_values):
        return self.feed_forward(x_values)
    
class MLPptorch(nn.Module):
    # как и раньше для инициализации на вход нужно подать размеры входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Sigmoid(),                    # функция активации
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid()
                                    
                                    # nn.Linear(in_size, hidden_size),
                                    # nn.ReLU(),                    # функция активации
                                    # nn.Linear(hidden_size, hidden_size),
                                    # nn.ReLU(),
                                    # nn.Linear(hidden_size, hidden_size),
                                    # nn.ReLU(),
                                    # nn.Linear(hidden_size, out_size),
                                    # nn.ReLU(),
        )

    
    # прямой проход    
    def forward(self,x):
        return self.layers(x)
    
# функция обучения
def train(x, y, num_iter):
    for i in range(0,num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%10==0:
           print('Достоверность на ' + str(i+1) + ' итерации: ', 1 - loss.item())
    return loss.item()
    
df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
X = df.iloc[0:100, 0:3].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

net = MLPptorch(inputSize,hiddenSizes,outputSize)
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

loss_ = train(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), 1000)

pred = net.forward(x)
# for name, param in net.named_parameters():
#     print(name, param)
