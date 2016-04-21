# -*- coding=utf8 -*-
"""
    a simple example to show how to user theano
"""
import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = 0.5 * x * x + x * T.sin(x)
dy = T.grad(y, x)

func = function(inputs=[x], outputs=dy)

start = 5.0
iter_num = 10000
rate = 0.01
for i in xrange(iter_num):
    d = func(start)
    if d < 0.005:
        break
    start -= rate * d

print start
