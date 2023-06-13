# -*- codeing = utf-8 -*-
# Time : 2023/6/13 14:34
# @Auther : zhouchao
# @File: spectrail.py
# @Software:PyCharm

import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from utils import read_rgb
import cv2


#%%

input_image = envi.open('data/hebing.hdr', 'data/hebing.raw')
#%%
view = imshow(input_image, (29, 19, 9))
print("1")