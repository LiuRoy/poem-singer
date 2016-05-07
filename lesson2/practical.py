# -*- coding=utf8 -*-
"""
    ANN神经网络分类示例
"""
import os
import csv

import numpy
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Activation

numpy.random.seed(50)  # 让每次运行结构都一样
current_dir = os.path.abspath(os.path.curdir)


class IrisData(object):
    """读取iris数据"""

    iris_path = os.path.join(current_dir, 'iris.data.csv')
    iris_type = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    @classmethod
    def load(cls):
        """
        读取csv数据
        :return numpy.ndarray
        """
        with open(cls.iris_path) as iris_file:
            reader = csv.reader(iris_file)

            input_x, input_y = [], []
            for line in reader:
                input_x.append(line[:-1])
                input_y.append(cls.iris_type[line[-1]])
            return numpy.array(input_x), numpy.array(input_y)


class LossHistory(Callback):
    """记录loss"""
    def __init__(self):
        Callback.__init__(self)
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        """每次epoch计算完loss后保存"""
        self.losses.append(logs.get('loss'))


class Model(object):
    """神经网络模型"""
    def __init__(self, n_inputs, n_classes, embedding_dims=2):
        """
        构造函数
        :param n_inputs: (int) 输入维度
        :param n_classes: (int) 输出类型个数
        :param embedding_dims: (int) 隐藏层维度
        """
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.model = Sequential()
        self.loss_history = LossHistory()

    def build_model(self):
        """
        参考practical4, 模型定义sigmoid -> softmax
        """
        self.model.add(Dense(self.embedding_dims, input_dim=self.n_inputs))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))

        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt, metrics=["accuracy"])

    def train(self, input_x, input_y):
        """
        训练数据及
        :param input_x: numpy.ndarray iris特征数据
        :param input_y: numpy.ndarray iris label数据
        """
        self.model.fit(input_x, input_y,
                       batch_size=32, nb_epoch=150, verbose=1,
                       callbacks=[self.loss_history])

    def predict(self, test_x, test_y):
        """
        训练数据及
        :param test_x: numpy.ndarray 测设iris特征数据
        :param test_y: numpy.ndarray iris label数据
        """
        return self.model.evaluate(test_x, test_y, verbose=0)

    def draw_loss(self):
        """画loss时序图"""
        x, y = [], []
        for counter, loss in enumerate(self.loss_history.losses):
            x.append(counter)
            y.append(loss)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(x, y, color="blue",linewidth=2)
        plt.title("loss")
        plt.show()


if __name__ == '__main__':
    iris_x, iris_y = IrisData.load()
    iris_y = to_categorical(iris_y).astype('int')

    model = Model(4, 3, 2)
    model.build_model()
    model.train(iris_x, iris_y)

    score = model.predict(iris_x, iris_y)
    print('Test accuracy:', score[1])

    model.draw_loss()
