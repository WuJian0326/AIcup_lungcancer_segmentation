import random
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
from tqdm.contrib import tzip
import os
import random
from tqdm import trange


class Image_Process():
    def __init__(self, image: np.array):
        self.image = image
        self.h, self.w = self.image.shape[0], self.image.shape[1]

    def image_scale_resize(self, out_h, out_w):
        h_scale = out_h / self.h
        w_scale = out_w / self.w

        out = round(self.w * min(h_scale, w_scale)), round(self.h * min(h_scale, w_scale))

        image = cv2.resize(self.image, out)
        h, w = image.shape[0], image.shape[1]
        center_h, center_w = (out_h - h) // 2, (out_w - w) // 2
        zh, zw = 0, 0
        if center_h * 2 != (out_h - h):
            zh = 1
        elif center_w * 2 != (out_w - w):
            zw = 1
        if len(image.shape) == 3:
            pading = np.pad(image, ((center_h, center_h + zh), (center_w, center_w + zw), (0, 0)), 'constant')
        else:
            pading = np.pad(image, ((center_h, center_h + zh), (center_w, center_w + zw)), 'constant')
        return pading

    def random_cut(self, image: np.array, mask: np.array, train_input_size, num=30):
        in_h, in_w, in_c = image.shape
        dis_h, dis_w = train_input_size
        image_list = []
        mask_list = []
        for i in range(num):
            h_start = random.randint(0, in_h - dis_h)
            w_start = random.randint(0, in_w - dis_w)
            img = image[h_start:h_start + dis_h, w_start:w_start + dis_w, :]
            mas = mask[h_start:h_start + dis_h, w_start:w_start + dis_w]

            image_list.append(img)
            mask_list.append(mas)
        return image_list, mask_list










class Segmentation_Image_Process():
    def __init__(self, Org_image_folder, Org_mask_folder, extention_image='jpg', extention_mask='jpg'):
        self.Org_image_folder = Org_image_folder
        self.Org_mask_folder = Org_mask_folder
        self.image_path = self.get_all_file_extension_path(self.Org_image_folder, extention_image)
        self.mask_path = self.get_all_file_extension_path(self.Org_mask_folder, extention_mask)

    def get_all_file_extension_path(self, path, file_extension='jpg'):
        file_extension = '*.' + file_extension
        path = os.path.join(path, file_extension)
        file_path = glob.glob(path)
        for i in range(len(file_path)):
            file_path[i] = file_path[i].replace('\\', '/')
        return file_path

    def create_folder(self, Save_Folder):
        try:
            os.makedirs(Save_Folder)
        except FileExistsError:
            print("檔案已存在。")

    def Training_data_folder(self, Save_Folder, Sub_Folder):
        self.create_folder(Save_Folder)
        sub = []
        for f in Sub_Folder:
            path = os.path.join(Save_Folder, f)
            self.create_folder(path)
            sub.append(path.replace('\\', '/'))
        return sub

    def save_array_image(self, image_list: list, path: str, name: str):
        for idx, image in enumerate(image_list):
            p = os.path.join(path, name + str(idx) + '.jpg')
            cv2.imwrite(p, image)

    def save_image(self, image, path: str, name: str):
        p = os.path.join(path, name + '.jpg')
        cv2.imwrite(p, image)

    def rate_resize(self, image, rate):
        image = cv2.resize(image, (round(image.shape[0] * rate), round(image.shape[1] * rate)))

    def random_resize(self, image, mask, input_size):
        h, w = image.shape[0], image.shape[1]
        r = random.uniform(0.3, 0.85)

        nh, nw = round((r * h)), round(r * w)
        img = cv2.resize(image, (nw, nh))
        mas = cv2.resize(mask, (nw, nh))
        mask_new = Image.new('L', (input_size[1], input_size[0]), 0)
        image_new = Image.new('RGB', (input_size[1], input_size[0]), (0, 0, 0))

        bh, bw = int(abs(nh - input_size[0])), int(abs(nw - input_size[1]))

        if nh > input_size[0] and nw > input_size[1]:
            py, px = random.randint(-bh, 0), random.randint(-bw, 0)

        else:
            py, px = random.randint(0, bh), random.randint(0, bw)

        image_new.paste(Image.fromarray(img), (px, py))
        image_new = np.array(image_new)
        mask_new.paste(Image.fromarray(mas), (px, py))
        mask_new = np.array(mask_new)

        return image_new, mask_new

    def distortion_resize(self, image, mask, input_size):
        h, w = image.shape[0], image.shape[1]
        siz = []
        r = random.uniform(-0.21,0.21)
        rate = h/w + r
        h = int(rate * w)
        siz.append([h, w])

        r = random.uniform(-0.5, 0.5)
        h, w = image.shape[0], image.shape[1]
        rate = w / h + r
        w = int(rate * h)
        siz.append([h, w])

        h_w = siz[random.randint(0,1)]
        image = cv2.resize(image, (h_w[1], h_w[0]))
        mask = cv2.resize(mask, (h_w[1], h_w[0]))
        IPG = Image_Process(image)
        image = IPG.image_scale_resize(input_size[0], input_size[1])
        IPM = Image_Process(mask)
        mask = IPM.image_scale_resize(input_size[0], input_size[1])

        return image, mask

    def read_all_image(self, Save_Folder, Sub_Folder, img_fig=4, random_choice=300, train_size=(471, 858)):
        Sub = self.Training_data_folder(Save_Folder, Sub_Folder)

        for idx, (img_path, msk_path) in enumerate(tzip(self.image_path, self.mask_path)):
            name = img_path.split('/')[-1].split('.')[0]
            self.image_list = []
            self.mask_list = []
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(msk_path))
            dimage, dmask = self.distortion_resize(image, mask, train_size)
            self.image_list.append(dimage)
            self.mask_list.append(dmask)

            for i in range(img_fig):
                rimage, rmask = self.random_resize(image, mask, input_size=train_size)
                self.image_list.append(rimage)
                self.mask_list.append(rmask)
            self.save_array_image(self.image_list, Sub[0], name)
            self.save_array_image(self.mask_list, Sub[1], name)



        self.image_list = []
        self.mask_list = []
        for i in trange(random_choice):
            image, mask = self.random_choice(train_size)
            self.image_list.append(image)
            self.mask_list.append(mask)

        self.save_array_image(self.image_list, Sub[0], 'miximage')
        self.save_array_image(self.mask_list, Sub[1], 'miximage')




    def random_choice(self, train_size):
        c = random.randint(1,2) * 2
        ls = []
        img_ls = []
        mas_ls = []
        for i in range(c):
            ls.append(random.randint(0, len(self.image_path) - 1))

        if c == 2:
            r = random.uniform(0.3, 0.7)
            left_w = int(r * train_size[1])
            right_w = train_size[1] - left_w
            blance = [left_w, right_w]

            for i in range(len(ls)):
                image = np.array(Image.open(self.image_path[ls[i]]))
                mask = np.array(Image.open(self.mask_path[ls[i]]))
                IP_Image = Image_Process(image)
                IP_Mask = Image_Process(mask)
                image = IP_Image.image_scale_resize(train_size[0], train_size[1])
                mask = IP_Mask.image_scale_resize(train_size[0], train_size[1])
                large = 0
                head = 0
                for j in range(0, train_size[1] - blance[i]):
                    if np.sum(mask[:, j:j+blance[i]]) > large:
                        large = np.sum(mask[:, j:j+blance[i]])
                        head = j

                img_ls.append(image[:, head:head+blance[i], :])
                mas_ls.append(mask[:, head:head+blance[i]])

            image = np.concatenate([img_ls[0], img_ls[1]], 1)
            mask = np.concatenate([mas_ls[0], mas_ls[1]], 1)
            return image, mask

        elif c == 4:
            left_w = int(train_size[1] * 0.5)
            right_w = train_size[1] - left_w
            up_h = int(train_size[0] * 0.5)
            down_h = train_size[0] - up_h

            blance_w = [left_w, left_w, right_w, right_w]
            blance_h = [up_h, down_h, up_h, down_h]

            for i in range(len(ls)):
                image = np.array(Image.open(self.image_path[ls[i]]))
                mask = np.array(Image.open(self.mask_path[ls[i]]))
                IP_Image = Image_Process(image)
                IP_Mask = Image_Process(mask)
                image = IP_Image.image_scale_resize(train_size[0], train_size[1])
                mask = IP_Mask.image_scale_resize(train_size[0], train_size[1])
                large = 0
                head_x = 0
                head_y = 0
                for x in range(0, train_size[1] - blance_w[i],10):
                    for y in range(0, train_size[0] - blance_h[i],10):
                        if np.sum(mask[y:y+blance_h[i], x:x+blance_w[i]]) > large:
                            large = np.sum(mask[y:y+blance_h[i], x:x+blance_w[i]])
                            head_x = x
                            head_y = y

                img_ls.append(image[head_y:head_y+blance_h[i], head_x:head_x+blance_w[i], :])
                mas_ls.append(mask[head_y:head_y+blance_h[i], head_x:head_x+blance_w[i]])

            image = np.concatenate([np.concatenate([img_ls[0], img_ls[1]], 0), np.concatenate([img_ls[2], img_ls[3]], 0)], 1)
            mask = np.concatenate([np.concatenate([mas_ls[0],mas_ls[1]], 0), np.concatenate([mas_ls[2], mas_ls[3]], 0)], 1)
            return image, mask





            # plt.subplot(2,1,1)
            # plt.imshow(image)
            # plt.subplot(2,1,2)
            # plt.imshow(mask)
            # plt.show()







if __name__ == '__main__':
    Org_Image = 'Origial_Image/Train_Images'
    Org_Mask = 'Origial_Image/Train_Mask'
    Save_Folder = 'Mix_858_471'
    Sub_Folder = ['Train_Images', 'Train_Mask']

    train_size = (471, 858)
    SIP = Segmentation_Image_Process(Org_Image, Org_Mask)
    SIP.read_all_image(Save_Folder, Sub_Folder, 4, 300, train_size)



    # for idx,(img_path, msk_path) in enumerate(tzip(image_path, mask_path)):
    #     image = np.array(Image.open(img_path))
    #     mask = np.array(Image.open(msk_path))
    #
    #     img_ls, mask_ls = Image_Process(image).random_cut(image, mask, (858,471))
    #
    #     break




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











