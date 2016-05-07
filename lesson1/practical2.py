# -*- coding=utf8 -*-
"""
    使用keras实现logistic分类器
"""
import os
import gzip
import urllib
import pickle

import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

current_dir = os.path.abspath(os.path.curdir)
numpy.random.seed(1337)  # 每一次运行结果都一样


class Mnist(object):
    """训练和测试数据"""
    path_remote = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    path_dataset = os.path.join(current_dir, 'mnist.pkl.gz')

    @classmethod
    def download(cls):
        """下载数据集"""
        if not os.path.exists(cls.path_dataset):
            print 'start download ...'
            urllib.urlretrieve(cls.path_remote, cls.path_dataset)
            print 'download finish.'

    @classmethod
    def load_dataset(cls):
        """读取数据文件"""
        cls.download()
        with gzip.open(cls.path_dataset, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        return train_set, valid_set, test_set


def create_model(input_dim, output_dim):
    """
    创建logistic模型
    :param input_dim: (int) 输入维度
    :param output_dim: (int) 输出维度
    :return: Sequential
    """
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])
    return model

if __name__ == '__main__':
    train_set, validate_set, test_set = Mnist.load_dataset()

    train_set_x, train_set_y = train_set
    validate_set_x, validate_set_y = validate_set
    test_set_x, test_set_y = test_set

    input_dim = test_set_x.shape[1]
    output_dim = test_set_y.max() - test_set_y.min() + 1

    train_set_y = to_categorical(train_set_y)
    validate_set_y = to_categorical(validate_set_y)
    test_set_y = to_categorical(test_set_y)

    model = create_model(input_dim, output_dim)
    model.fit(train_set_x, train_set_y,
              batch_size=256, nb_epoch=50, verbose=1,
              validation_data=(validate_set_x, validate_set_y))
    score = model.evaluate(test_set_x, test_set_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
