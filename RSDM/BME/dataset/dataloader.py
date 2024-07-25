from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

torch.manual_seed(3)

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data

class Normalize(object):
    def __init__(self, mag_norm, ZeroToOne=True):
        super(Normalize, self).__init__()
        self.ZeroToOne = ZeroToOne
        self.num = 0 if ZeroToOne else 0.5
        self.mag_norm = mag_norm

    def __call__(self, data):
        for key in data.keys():
            if key == 'blur_mag':
                # v2
                # data[key] = (data[key] / self.mag_norm).copy()
                # data[key][data[key] > 1] = 1
                # data[key] = (data[key] - self.num).copy()

                # v1
                data[key] = ((data[key] / self.mag_norm) - self.num).copy()
            else:
                data[key] = ((data[key] / 255) - self.num).copy()
        return data

class ToTensor(object):
    def __call__(self, data):
        for key in data.keys():
            if key == 'blur_img':
                data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()
            else:
                data[key] = torch.from_numpy(data[key]).clone()
        return data

class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("resize error")

    def __call__(self, data):
        data['blur_img'] = data['blur_img'].resize(self.size, resample=Image.BILINEAR).copy()
        return data

class BlurMagDataset(Dataset):
    def __init__(self,dataset_root,train=True):
        
        self.train = train
        self.crop_size = 256
        self.mag_norm = 205 # the max magnitude of training dataset
        if train:
            self.transform = transforms.Compose([RandomCrop(self.crop_size, self.crop_size), Normalize(self.mag_norm), ToTensor()])
            self.blur_img_path_list = []
            self.blur_mag_path_list = []
            dataset_list = os.listdir(dataset_root)
            
            for dataset in dataset_list:
                video_list = os.listdir(os.path.join(dataset_root, dataset))
                for video in video_list:
                    file_list = os.listdir(os.path.join(dataset_root, dataset, video, 'blur_img'))
                    for file in file_list:
                        self.blur_img_path_list.append(os.path.join(dataset_root, dataset, video, 'blur_img', file))
                        self.blur_mag_path_list.append(os.path.join(dataset_root, dataset, video, 'blur_mag_np', file.replace('png','npy')))

        else:
            self.transform = transforms.Compose([Normalize(self.mag_norm), ToTensor()])
            self.blur_img_path_list = []
            file_list = os.listdir(os.path.join(dataset_root, 'Blur/RGB'))
            for file in file_list:
                self.blur_img_path_list.append(os.path.join(dataset_root, 'Blur/RGB', file))

        self.length = len(self.blur_img_path_list)
                
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if self.train:
            blur_img = cv2.imread(self.blur_img_path_list[idx]).astype(np.float32)
            blur_mag = np.load(self.blur_mag_path_list[idx])
            sample = {
                'blur_img': blur_img,
                'blur_mag': blur_mag
            }
            self.transform(sample)
            return sample['blur_img'], sample['blur_mag']
        else:

            blur_img = cv2.imread(self.blur_img_path_list[idx]).astype(np.float32)
            sample = {
                'blur_img': blur_img,
            }
            self.transform(sample)
            file_name = self.blur_img_path_list[idx].split('/')[-1]
            video_name = self.blur_img_path_list[idx].split('/')[-3]

            return sample['blur_img'], video_name, file_name
            
if __name__ == "__main__":
    train_path = "disk2/jthe/datasets/GOPRO_blur_magnitude/train"

    blurmagdataste = BlurMagDataset(dataset_root=train_path, train=True)
    print(len(blurmagdataste))
    blur,mmp = blurmagdataste[0]
    blur,mmp = blurmagdataste[20]
    blur,mmp = blurmagdataste[1000]
    print(blur.shape)
    print(mmp.shape)
    print(mmp)