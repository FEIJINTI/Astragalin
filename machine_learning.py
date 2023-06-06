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




def DecisionTree(train_x, train_y, test_x, test_y):
    dt = DecisionTreeClassifier(random_state=65)
    dt = dt.fit(train_x, train_y)
    t1 = time.time()
    y_pred = dt.predict(test_x)
    print("预测时间：", time.time()-t1)
    print("DT训练模型评分：" + str(accuracy_score(train_y, dt.predict(train_x))))
    print("DT待测模型评分：" + str(accuracy_score(test_y, dt.predict(test_x))))
    print('DT预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('DT分类报告：' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('DT混淆矩阵：' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少

