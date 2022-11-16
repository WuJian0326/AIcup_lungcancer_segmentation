import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


def get_train_transform():
    transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightness(limit=0.1, always_apply=False, p=0.5),
            A.RandomContrast(limit=0.1, always_apply=False, p=0.5),
        ], p=0.3),
        # A.OneOf([
        #     A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     A.GridDistortion(p=0.3, num_steps=10),
        #     A.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5)
        # ]),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
            ),
        ToTensorV2(),
    ])
    return transform

def get_vaild_transform():
    transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
            ),
        ToTensorV2(),
    ])
    return transform

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])#.replace('.jpg','.png')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.
        mask[mask > 0.8] = 1.0


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask






