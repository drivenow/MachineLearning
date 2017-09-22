# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:20:18 2017

@author: Administrator
"""

import numpy as np
from keras import backend as K
from keras.utils.generic_utils import get_from_module

def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)
"""
shape参数何来？
convultion：初始化在build函数中；W_shape(nb_filter,stack_size(feature_map_num),dim1,dim2,dmi3)
这个shape传递到滤波器初始化函数中，即初始化这么多滤波器？
故返回：fin：input_feature_map*dim1*dim2
        fout：output_feature_map*dim1*dim2
"""
def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # Assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid dim_ordering: ' + dim_ordering)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))#返回各元素的乘积
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(shape, scale=0.05, name=None, dim_ordering='th'):
    return K.random_uniform_variable(shape, -scale, scale, name=name)

#产生正太分布的tensor对象，均值0，方差0.05
def normal(shape, scale=0.05, name=None, dim_ordering='th'):
    return K.random_normal_variable(shape, 0.0, scale, name=name)


    
def xavier_normal(shape, name=None, dim_ordering='th'):
    """He normal variance scaling initializer.

    # References
        He et al., http://arxiv.org/abs/1502.01852
    """
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = 1.0/fan_in
    return normal(shape, s, name=name)
