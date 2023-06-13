# -*- codeing = utf-8 -*-
# Time : 2023/6/5 14:25
# @Auther : zhouchao
# @File: machine_learning.py
# @Software:PyCharm
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def DecisionTree(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    dt = DecisionTreeClassifier(random_state=65, class_weight=class_weight)
    dt = dt.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(dt, f)

    t1 = time.time()

    y_pred = dt.predict(test_x)
    print("预测时间：", time.time()-t1)
    print("DT训练模型评分：" + str(accuracy_score(train_y, dt.predict(train_x))))
    print("DT待测模型评分：" + str(accuracy_score(test_y, dt.predict(test_x))))
    print('DT预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('DT分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('DT混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少


def svm_classifier(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    svm = SVC(kernel='linear', random_state=65, class_weight=class_weight)
    svm.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(svm, f)

    t1 = time.time()

    y_pred = svm.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("SVM训练模型评分：" + str(accuracy_score(train_y, svm.predict(train_x))))
    print("SVM待测模型评分：" + str(accuracy_score(test_y, svm.predict(test_x))))
    print('SVM预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('SVM分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('SVM混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少


def RandomForest(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    rf = RandomForestClassifier(n_estimators= 2, random_state=65, class_weight=class_weight)
    rf = rf.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(rf, f)

    t1 = time.time()

    y_pred = rf.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("RF训练模型评分：" + str(accuracy_score(train_y, rf.predict(train_x))))
    print("RF待测模型评分：" + str(accuracy_score(test_y, rf.predict(test_x))))
    print('RF预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('RF分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('RF混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少



def AdaBoost(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    ada = AdaBoostClassifier(n_estimators= 2, random_state=65, class_weight=class_weight)
    ada = ada.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(ada, f)

    t1 = time.time()

    y_pred = ada.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("AdaBoost训练模型评分：" + str(accuracy_score(train_y, ada.predict(train_x))))
    print("AdaBoost待测模型评分：" + str(accuracy_score(test_y, ada.predict(test_x))))
    print('AdaBoost预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('AdaBoost分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('AdaBoost混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少
    

def GradientBoosting(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    gb = GradientBoostingClassifier(n_estimators= 2, random_state=65, class_weight=class_weight)
    gb = gb.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(gb, f)

    t1 = time.time()

    y_pred = gb.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("GradientBoosting训练模型评分：" + str(accuracy_score(train_y, gb.predict(train_x))))
    print("GradientBoosting待测模型评分：" + str(accuracy_score(test_y, gb.predict(test_x))))
    print('GradientBoosting预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('GradientBoosting分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('GradientBoosting混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少


def XGBoost(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    xgb = XGBClassifier(n_estimators= 2, random_state=65, class_weight=class_weight)
    xgb = xgb.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(xgb, f)

    t1 = time.time()

    y_pred = xgb.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("XGBoost训练模型评分：" + str(accuracy_score(train_y, xgb.predict(train_x))))
    print("XGBoost待测模型评分：" + str(accuracy_score(test_y, xgb.predict(test_x))))
    print('XGBoost预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('XGBoost分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('XGBoost混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少


def MLP(train_x, train_y, test_x, test_y, file_name, class_weight=None):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, alpha=0.0001, solver='sgd',
                        verbose=10, random_state=65, tol=0.000000001, class_weight=class_weight)
    mlp = mlp.fit(train_x, train_y)
    # 保存模型
    with open(file_name, 'wb') as f:
        pickle.dump(mlp, f)

    t1 = time.time()

    y_pred = mlp.predict(test_x)
    print("预测时间：", time.time() - t1)
    print("MLP训练模型评分：" + str(accuracy_score(train_y, mlp.predict(train_x))))
    print("MLP待测模型评分：" + str(accuracy_score(test_y, mlp.predict(test_x))))
    print('MLP预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('MLP分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('MLP混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少





def predict(x, file_name):
    with open(file_name, 'rb') as f:
        dt = pickle.load(f)
    t1 = time.time()
    y_pred = dt.predict(x)
    print("预测时间：", time.time() - t1)
    return y_pred