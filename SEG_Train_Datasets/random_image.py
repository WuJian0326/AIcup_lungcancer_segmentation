import random
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from Image_Process import *
from tqdm.contrib import tzip
import os
import random
import math

def rate_resize(image, mask, input_size):
    h, w = image.shape[0], image.shape[1]
    rate = np.random.uniform(low=0.2, high=0.8, size=10).tolist()
    out_image = []
    out_mask = []
    for r in rate:
        nh, nw = round((r * h)), round(r * w)
        img = cv2.resize(image, (nw, nh))
        mas = cv2.resize(mask, (nw, nh))
        mask_new = Image.new('L', (input_size[1], input_size[0]), 0)
        image_new = Image.new('RGB', (input_size[1], input_size[0]), (0, 0, 0))

        bh, bw = abs(nh - input_size[0]), abs(nw - input_size[1])

        if r >= 0.5:
            py, px = random.randint(-bh, 0), random.randint(-bw, 0)
        else:
            py, px = random.randint(0, bh), random.randint(0, bw)

        image_new.paste(Image.fromarray(img), (px, py))
        out_image.append(np.array(image_new))
        mask_new.paste(Image.fromarray(mas), (px, py))
        out_mask.append(np.array(mask_new))

    return out_image, out_mask





Org_Image = 'Origial_Image/Train_Images'
Org_Mask = 'Origial_Image/Train_Mask'

Save_Folder = 'Random_image_858_471/'
Sub_Folder = ['Train_Images', 'Train_Mask']

Training_data_folder(Save_Folder, Sub_Folder)
input_size = (471, 858)
image_path = get_all_file_extension_path(Org_Image, 'jpg')
mask_path = get_all_file_extension_path(Org_Mask, 'jpg')




for idx, (img_path, msk_path) in enumerate(tzip(image_path, mask_path)):
    image = np.array(Image.open(img_path))
    mask = np.array(Image.open(msk_path))
    img_ls, msk_ls = rate_resize(image, mask, input_size)

    for i in range(len(img_ls)):
        save_image(img_ls[i], Save_Folder + Sub_Folder[0] + '/', img_path.split('/')[-1].split('.')[-2] + '_' + str(i))
    for i in range(len(msk_ls)):
        save_image(msk_ls[i], Save_Folder + Sub_Folder[1] + '/', img_path.split('/')[-1].split('.')[-2] + '_' + str(i))






