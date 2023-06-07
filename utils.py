# -*- codeing = utf-8 -*-
# Time : 2023/6/5 9:41
# @Auther : zhouchao
# @File: utils.py
# @Software:PyCharm
import cv2

import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def read_envi_ascii(file_name, save_xy=False, hdr_file_name=None):
    """
    Read envi ascii file. Use ENVI ROI Tool -> File -> output ROIs to ASCII...

    :param file_name: file name of ENVI ascii file
    :param hdr_file_name: hdr file name for a "BANDS" vector in the output
    :param save_xy: save the x, y position on the first two cols of the result vector
    :return: dict {class_name: vector, ...}
    """
    number_line_start_with = "; Number of ROIs: "
    roi_name_start_with, roi_npts_start_with = "; ROI name: ", "; ROI npts:"
    data_start_with, data_start_with2, data_start_with3 = ";    ID", ";   ID", ";     ID"
    class_num, class_names, class_nums, vectors = 0, [], [], []
    with open(file_name, 'r') as f:
        for line_text in f:
            if line_text.startswith(number_line_start_with):
                class_num = int(line_text[len(number_line_start_with):])
            elif line_text.startswith(roi_name_start_with):
                class_names.append(line_text[len(roi_name_start_with):-1])
            elif line_text.startswith(roi_npts_start_with):
                class_nums.append(int(line_text[len(roi_name_start_with):-1]))
            elif line_text.startswith(data_start_with) or line_text.startswith(data_start_with2) or line_text.startswith(data_start_with3):
                col_list = list(filter(None, line_text[1:].split(" ")))
                assert (len(class_names) == class_num) and (len(class_names) == len(class_nums))
                break
            elif line_text.startswith(";"):
                continue
        for vector_rows in class_nums:
            vector_str = ''
            for i in range(vector_rows):
                vector_str += f.readline()
            vector = np.fromstring(vector_str, dtype=float, sep=" ").reshape(-1, len(col_list))
            assert vector.shape[0] == vector_rows
            vector = vector[:, 3:] if not save_xy else vector[:, 1:]
            vectors.append(vector)
            f.readline()  # suppose to read a blank line
    if hdr_file_name is not None:
        import re
        with open(hdr_file_name, 'r') as f:
            hdr_info = f.read()
        bands = re.findall(r"wavelength = {[^{}]+}", hdr_info, flags=re.IGNORECASE | re.MULTILINE)
        bands_num = re.findall(r"bands\s*=\s*(\d+)", hdr_info, flags=re.I)
        if (len(bands) == 0) or len(bands_num) == 0:
            Warning("The given hdr file is invalid, can't find bands = ? or wavelength = {?}.")
        else:
            bands = re.findall(r'{[^{}]+}', bands[0], flags=re.MULTILINE)[0][3:-2]
            bands = bands.split(',\n')
            bands = np.asarray(bands, dtype=float)
            bands_num = int(bands_num[0])
            if bands_num == bands.shape[0]:
                bands = np.array(bands, dtype=float)
                vectors.append(bands)
                class_names.append("BANDS")
            else:
                Warning("The given hdr file is invalid, bands num is not equal to wavelength.")
    return dict(zip(class_names, vectors))


def ga_feature_extraction(data_x, data_y):
    '''
    使用遗传算法进行特征提取
    :param data_x: 特征
    :param data_y: 类别
    '''
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x, data_y, test_size=0.3)
    clf = DecisionTreeClassifier(random_state=3)
    selector = GeneticSelectionCV(clf, cv=20,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=7,
                                  n_population=200,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=200,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(Xtrain, Ytrain)
    Xtrain_ga, Xtest_ga = Xtrain[:, selector.support_], Xtest[:, selector.support_]
    clf = clf.fit(Xtrain_ga, Ytrain)
    print(np.where(selector.support_ == True))
    y_pred = clf.predict(Xtest_ga)
    print(classification_report(Ytest, y_pred))
    print(confusion_matrix(Ytest, y_pred))


def read_raw(file_name, shape=None,  setect_bands=None):
    '''
    读取raw文件
    :param file_name: 文件名
    :param setect_bands: 选择的波段
    :return: 波段数据
    '''
    if shape is None:
        shape = (692, 272, 384)
    with open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape).transpose(0, 2, 1)
    if setect_bands is not None:
        data = data[:, :, setect_bands]
    return data


def save_raw(file_name, data):
    '''
    保存raw文件
    :param file_name: 文件名
    :param data: 数据
    '''
    data = data.transpose(0, 2, 1)
    # 将data转换为一维数组
    data = data.reshape(-1)
    with open(file_name, 'wb') as f:
        f.write(data.astype(np.float32).tobytes())