import random
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tqdm.contrib import tzip
import os

def image_scale_resize(image, out_h, out_w):
    h, w = image.shape[0], image.shape[1]
    h_scale = out_h / h
    w_scale = out_w / w

    out = round(w * min(h_scale, w_scale)), round(h * min(h_scale, w_scale))

    image = cv2.resize(image, out)
    h, w = image.shape[0], image.shape[1]
    center_h, center_w = (out_h-h)//2, (out_w-w) // 2
    zh, zw = 0, 0
    if center_h * 2 != (out_h-h):
        zh = 1
    elif center_w * 2 != (out_w-w):
        zw = 1
    if len(image.shape) == 3:
        pading = np.pad(image,((center_h, center_h + zh),(center_w, center_w+ zw),(0, 0)), 'constant')
    else:
        pading = np.pad(image, ((center_h, center_h + zh), (center_w, center_w + zw)), 'constant')
    return pading

def get_all_file_extension_path(path, file_extension='jpg'):
    file_extension = '*.' + file_extension
    path = os.path.join(path,file_extension)
    file_path = glob.glob(convert_path(path))
    for i in range(len(file_path)):
        file_path[i] = convert_path(file_path[i])
    return file_path

def fill_the_image(image:np.array, train_input_size):
    h, w = image.shape[0], image.shape[1]
    d_h, d_w = h % train_input_size[0], w % train_input_size[1]

    if d_h != 0:
        d_h = train_input_size[0] - d_h
    if d_w != 0:
        d_w = train_input_size[1] - d_w
    image = Image.fromarray(image)
    black = Image.new('RGB',(w + d_w, h + d_h), (0,0,0))
    black.paste(image, (0, 0))
    return np.array(black)

def random_cut(image:np.array, mask:np.array, train_input_size, num=30):
    in_h, in_w, in_c = image.shape
    dis_h, dis_w = train_input_size
    image_list = []
    mask_list = []
    for i in range(num):
        h_start = random.randint(0, in_h-dis_h)
        w_start = random.randint(0, in_w-dis_w)
        img = image[h_start:h_start + dis_h, w_start:w_start + dis_w, :]
        mas = mask[h_start:h_start + dis_h, w_start:w_start + dis_w, :]

        image_list.append(img)
        mask_list.append(mas)
    return image_list, mask_list

def cut_image(image:np.array, train_input_size):
    image = fill_the_image(image, train_input_size)
    in_h, in_w, in_c = image.shape
    dis_h, dis_w = train_input_size
    image_list = []

    for h in range(0, in_h, dis_h):
        for w in range(0, in_w, dis_w):
            img = image[h:h+dis_h, w:w+dis_w,:]
            image_list.append(img)

    return image_list, image

def save_array_image(image_list:list, path:str, name:str):
    for idx, image in enumerate(image_list):
        img = image[:,:,::-1]
        p = path + name + str(idx) + '.jpg'
        cv2.imwrite(p, img)

def save_image(image:np.array, path:str, name:str):
    p = path + name + '.jpg'
    cv2.imwrite(p, image)

def convert_path(path: str) -> str:
    return path.replace('\\', '/')

def create_folder(Save_Folder):
    try:
        os.makedirs(Save_Folder)
    except FileExistsError:
        print("檔案已存在。")

def Training_data_folder(Save_Folder, Sub_Folder):
    create_folder(Save_Folder)
    for f in Sub_Folder:
        path = os.path.join(Save_Folder, f)
        create_folder(path)


if __name__ == '__main__':
    Org_Image = 'Origial_Image/Train_Images'
    Org_Mask = 'Origial_Image/Train_Mask'

    Save_Folder = 'Size_515_283'
    Sub_Folder = ['Train_Images', 'Train_Mask']

    Training_data_folder(Save_Folder, Sub_Folder)


    input_size = (283,515)
    image_path = get_all_file_extension_path(Org_Image, 'jpg')
    mask_path = get_all_file_extension_path(Org_Mask, 'jpg')



    for idx,(img_path, msk_path) in enumerate(tzip(image_path, mask_path)):
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(msk_path))

        h, w = image.shape[0], image.shape[1]
        image = image_scale_resize(image, input_size[0], input_size[1])
        mask = image_scale_resize(mask, input_size[0], input_size[1])

        save_image(image, Save_Folder + '/Train_Images/', img_path.split('/')[-1].split('.')[0])
        save_image(mask, Save_Folder + '/Train_Mask/', msk_path.split('/')[-1].split('.')[0])





# for idx,(img_path, csv_path) in enumerate(zip(vaild_input, vaild_target)):
#     image = plt.imread(img_path)
#     h, w, c = image.shape
#     mask = create_mask_image(csv_path, h, w, c)
#
#     image_list, image = cut_image(image, input_size)
#     mask_list, mask = cut_image(mask, input_size)
#
#     i_list, m_list = random_cut(image, mask, input_size)
#     image_list = image_list + i_list
#     mask_list = mask_list + m_list
#
#     save_array_image(image_list,vaild_image_save, 'cut_' + str(idx) + '_')
#     save_array_image(mask_list, vaild_target_save, 'cut_' + str(idx) + '_')












