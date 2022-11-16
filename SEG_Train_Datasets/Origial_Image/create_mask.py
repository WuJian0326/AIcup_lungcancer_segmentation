import os
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Annotations_path = 'Train_Annotations'   #json位置
Mask_save_path = 'Train_Mask'            #遮罩儲存位置

try:
  os.makedirs(Mask_save_path)
except FileExistsError:
  print("檔案已存在。")

json_names = tqdm(os.listdir(Annotations_path))

for n in json_names:
    path = os.path.join(Annotations_path, n)

    with open(path) as f:
        data = json.load(f)

    h, w = data['imageHeight'], data['imageWidth']
    background = np.zeros((h,w))
    for shape in data['shapes']:
        points = shape['points']
        for i in range(len(points)):
            points[i] = [round(points[i][0]), round(points[i][1])]
        points = np.array([points],dtype=np.int32)
        cv2.fillPoly(background, points, (255,255,255))

    cv2.imwrite(os.path.join(Mask_save_path, n.split('.')[0] + '.jpg'), background)



