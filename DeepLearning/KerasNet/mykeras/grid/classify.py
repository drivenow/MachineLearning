# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:56:15 2017

@author: Administrator
"""

"""keras对sklearn的包装类，可以用grid_search参数
参数传入的顺序：
（1）首先是 fit, predict, predict_proba, and score函数中的参数
（2）其次是KerasClassifier中定义的参数
（3）再次是默认参数
注意： grid_search的默认score是estimator的score

param_grid:
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizer = ['Adadelta','RMSprop']
    loss=['categorical_crossentropy']
    batch_size=[10]
    
    param_grid = dict(optimizer=optimizer,
                      loss=loss,
                      batch_size=batch_size)

cv: 如果是多分类问题，默认是StrategyShuffle,其他都是K-cv

注意：只能用于预测标签
    The model is not configured to compute accuracy. 
    You should pass `metrics=["accuracy"]` to the `model.compile()
                                        
"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV


def grid_search_classify(model, X_data, Y_data, nb_epoch, param_grid,cv):
    AE_keras=KerasClassifier(build_fn=model, nb_epoch=nb_epoch, verbose=1)
    
    grid_search = GridSearchCV(estimator=AE_keras, param_grid=param_grid, n_jobs=1,cv=cv)#scoring=make_scorer(mean_squared_error)

    grid_result=grid_search.fit(X_data,Y_data)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(grid_result.grid_scores_)
    
    return grid_result

