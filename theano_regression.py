import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))      
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        #shape只有一个参数时，可以当做行，也可以当做列
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


# Make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#加入新坐标，目的在定义一个300行的列向量
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise        # y = x^2 - 0.5

# show the fake data
#plt.scatter(x_data, y_data)
#plt.show()

# determine the inputs dtype
x = T.dmatrix("x")
y = T.dmatrix("y")

# add layers
l1 = Layer(x, 1, 10, T.nnet.relu)
#定义了激励函数
l2 = Layer(l1.outputs, 10, 1, None)
#定义层时注意，前一个输出神经元个数必须等于后一个输入神经元个数，即10 = 10

# compute the cost
cost = T.mean(T.square(l2.outputs - y))

# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])
#因为cost和四个参数有关，分别对四个参数求偏导，得到梯度

# apply gradient descent
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)])
#每训练一次更新四个参数值

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
#让plt.show()不再阻塞
plt.show()

# prediction
predict = theano.function(inputs=[x], outputs=l2.outputs)

for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        print('经过第'+str(i)+'次训练:cost='+str(err))
        try:
            ax.lines.remove(lines[0])
            #移去上次学习的曲线
        except Exception:
            pass
        
        predict_value = predict(x_data)
        lines = ax.plot(x_data,predict_value,'r-')
        plt.pause(0.1)









        
