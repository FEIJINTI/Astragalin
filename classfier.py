
import logging
import sys
from typing import Optional

import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import ndimage
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import binom
import matplotlib.pyplot as plt
import time
import pickle
import os
import utils

class Astragalin(object):
    def __init__(self, load_from=None, debug_mode=False, class_weight=None):
        if load_from is None:
            self.model = DecisionTreeClassifier(random_state=65, class_weight=class_weight)
            self.log = utils.Logger(is_to_file=debug_mode)
            self.debug_mode = debug_mode







    def fit(self, data_x, data_y, test_size=0.3):
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, stratify=data_y)
        self.model.fit(train_x, train_y)
        y_pred = self.model.predict(test_x)
        print(confusion_matrix(test_y, y_pred))

        pre_score = accuracy_score(test_y, y_pred)
        self.log.log("Test accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(train_x)

        pre_score = accuracy_score(train_y, y_pred)
        self.log.log("Train accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(data_x)


        pre_score = accuracy_score(data_y, y_pred)
        self.log.log("Total accuracy is:" + str(pre_score * 100) + "%.")

        # 显示结果报告

        return int(pre_score * 100)

    def predict(self, data_x, ):
        select = SelectKBest(chi2, k=10)
        x_new = select.fit_transform(data_x, data_y)
        selected_features = select.get_support(indices=True)
        data_x = read_raw('data/01newrawfile_ref.raw', shape=(750, 288, 384), setect_bands=selected_features)
        # %%
        data_x_shape = data_x.shape
        data_x = data_x.reshape(-1, data_x.shape[2])
        data_y = self.model.predict(data_x)
        # %%
        data_y = data_y.reshape(data_x_shape[0], data_x_shape[1]).astype(np.uint8)
        # %%
        pre_data_y = np.zeros((data_y.shape[0], data_y.shape[1], 3), dtype=np.uint8)

        pre_data_y[data_y == 0] = [0, 0, 0]  # 黑色
        pre_data_y[data_y == 1] = [255, 0, 0]  # 红色
        pre_data_y[data_y == 2] = [0, 255, 0]  # 绿色
        pre_data_y[data_y == 3] = [0, 0, 255]  # 蓝色
        pre_data_y[data_y == 4] = [255, 255, 0]  # 黄色
        pre_data_y[data_y == 5] = [255, 0, 255]  # 紫色


    def connect_space(self, data_y):
        labels, num_features = ndimage.label(data_y)
        for i in range(1, num_features + 1):
            mask = (labels == i)
            counts = np.bincount(data_y[mask])
            # 如果2的个数在所有的像素点中占比超过25%，则认为是2，否则认为0和1中最多的是哪个就是哪个
            # 如果有count[2]，才进入判断，否则直接认为是0或者1
            if len(counts) > 2 and counts[2] / np.sum(counts) > 0.20:
                data_y[mask] = 2
            else:
                data_y[mask] = np.argmax(counts)
            data_y[mask] = np.argmax(counts)










    # def DecisionTree(self,train_x, train_y, test_x, test_y, file_name, class_weight=None):
    #     dt = DecisionTreeClassifier(random_state=65, class_weight=class_weight)
    #     dt = dt.fit(train_x, train_y)
    #     # 保存模型
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(dt, f)
    #
    #     t1 = time.time()
    #
    #     y_pred = dt.predict(test_x)
    #
    # def predict(x, file_name):
    #     with open(file_name, 'rb') as f:
    #         dt = pickle.load(f)
    #     t1 = time.time()
    #     y_pred = dt.predict(x)
    #     print("预测时间：", time.time() - t1)
    #     return y_pred
