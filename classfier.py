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
        else:
            self.lode(load_from)

    def lode(self, path=None):
        pass


    def fit(self, data_x, data_y):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=65)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        print(confusion_matrix(y_test, y_pred))

        pre_score = accuracy_score(y_test, y_pred)
        self.log.log("Test accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(x_train)

        pre_score = accuracy_score(y_train, y_pred)
        self.log.log("Train accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(data_x)

        pre_score = accuracy_score(data_y, y_pred)
        self.log.log("Total accuracy is:" + str(pre_score * 100) + "%.")

        return int(pre_score * 100)


    def fit_value(self, data_path='data/1.txt', select_bands=[91, 92, 93, 94, 95, 96, 97, 98, 99, 100]):
        data_x, data_y = self.data_construction(data_path, select_bands)
        self.fit(data_x, data_y)



    def data_construction(self, data_path, select_bands):
        data = utils.read_envi_ascii(data_path)
        beijing = data['beijing'][:, select_bands]
        zazhi1 = data['zazhi1'][:, select_bands]
        zazhi2 = data['zazhi2'][:, select_bands]
        huangqi = data['huangqi'][:, select_bands]
        gancaopian = data['gancaopian'][:, select_bands]
        hongqi = data['hongqi'][:, select_bands]
        beijing_y = np.zeros(beijing.shape[0])
        zazhi1_y = np.ones(zazhi1.shape[0])
        zazhi2_y = np.ones(zazhi2.shape[0]) * 2
        huangqi_y = np.ones(huangqi.shape[0]) * 3
        gancaopian_y = np.ones(gancaopian.shape[0]) * 4
        hongqi_y = np.ones(hongqi.shape[0]) * 5
        data_x = np.concatenate((beijing, zazhi1, zazhi2, huangqi, gancaopian, hongqi), axis=0)
        data_y = np.concatenate((beijing_y, zazhi1_y, zazhi2_y, huangqi_y, gancaopian_y, hongqi_y), axis=0)
        return data_x, data_y

    def predict(self, data_x):
        '''
        对数据进行预测
        :param data_x: 波段选择后的数据
        :return: 预测结果二值化后的数据，0为背景，1为杂质1,2为杂质2，3为黄芪，4为甘草片，5为红芪
        '''
        data_x_shape = data_x.shape
        data_x = data_x.reshape(-1, data_x.shape[2])
        data_y = self.model.predict(data_x)
        data_y = data_y.reshape(data_x_shape[0], data_x_shape[1]).astype(np.uint8)
        return data_y


# 连通域处理离散点
    def connect_space(self, data_y):
        labels, num_features = ndimage.label(data_y)
        for i in range(1, num_features + 1):
            mask = (labels == i)
            counts = np.bincount(data_y[mask])
            data_y[mask] = np.argmax(counts)
        return data_y


