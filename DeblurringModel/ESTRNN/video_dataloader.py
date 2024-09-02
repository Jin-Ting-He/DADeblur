#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random
import json

def rotation_matrix(angle):
    # 将角度转换为弧度
    rad = np.radians(angle)
    
    # 構建旋转矩阵
    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)
    rot_matrix = np.array([[cos_theta, -sin_theta],
                           [sin_theta, cos_theta]])
    
    return rot_matrix
def generate_neighbors_filenames(input_filename):
    base_name, extension = input_filename.split('.')
    number_part = base_name[-8:]
    
    number = int(number_part)
    
    adjacent_numbers = [number - 2, number - 1, number, number + 1, number + 2]
    adjacent_filenames = [f"{num:08d}.{extension}" for num in adjacent_numbers]
    #adjacent_filenames = [f"{num}.{extension}" for num in adjacent_numbers]
    return adjacent_filenames

def generate_neighbors_filenames_RealBlur(input_filename):
    base_name, extension = input_filename.split('.')
    number_part = base_name
    
    number = int(number_part)
    
    adjacent_numbers = [number - 2, number - 1, number, number + 1, number + 2]
    adjacent_filenames = [f"{num}.{extension}" for num in adjacent_numbers]

    return adjacent_filenames

def get_neighbor_imgs_json(file_path):
    video_idx = file_path.split('/')[-3]
    file_name = file_path.split('/')[-1]
    dataset_name = file_path.split('/')[-5]
    path_parts = file_path.split('/')
    # neighbors_root = '/'.join(path_parts[:-1]).replace('reblur','ori')
    neighbors_root = os.path.join("4TB/jthe/datasets",dataset_name, "test", video_idx, "Blur/RGB")
    # neighbors_root = '/'.join(path_parts[:-1])
    neighbors_list = generate_neighbors_filenames(file_name)
    neighbors_idx = [0,1,3,4]
    for idx in neighbors_idx:
        neighbors_list[idx] = os.path.join(neighbors_root, neighbors_list[idx])
    neighbors_list[2] = file_path
    return neighbors_list

def get_neighbor_imgs_json_RealBlur(file_path):
    video_idx = file_path.split('/')[-3]
    file_name = file_path.split('/')[-1]
    dataset_name = file_path.split('/')[-5]
    path_parts = file_path.split('/')
    neighbors_root = '/'.join(path_parts[:-1]).replace('reblur','ori')
    neighbors_list = generate_neighbors_filenames_RealBlur(file_name)
    neighbors_idx = [0,1,3,4]
    for idx in neighbors_idx:
        neighbors_list[idx] = os.path.join(neighbors_root, neighbors_list[idx])
    neighbors_list[2] = file_path
    return neighbors_list

def get_neighbor_imgs(file_path):
    video_idx = file_path.split('/')[-1].split('_')[0]
    file_name = file_path.split('/')[-1].split('_')[-1]
    # for 1ms8ms it would be [0:10]
    # for 2ms16ms and 3ms24ms it would be [0:11]
    dataset_name = file_path.split('/')[1][0:11]
    neighbors_root = os.path.join("dataset",dataset_name,"test",video_idx,"Blur/RGB")
    neighbors_list = generate_neighbors_filenames(file_name)
    neighbors_idx = [0,1,3,4]
    for idx in neighbors_idx:
        neighbors_list[idx] = os.path.join(neighbors_root, neighbors_list[idx])
    neighbors_list[2] = file_path
    return neighbors_list
class RandomRotate(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            if key != 'flow':
                data[key] = np.rot90(data[key], dirct).copy()
            else:
                vectors = data[key][:, : ,:2].copy()
                
                vectors_origin_shape = vectors.shape
                vectors = vectors.reshape((-1, 2))
                rot_matrix = rotation_matrix(90 * dirct)
                
                # 使用旋轉矩陣
                rotated_vectors = (rot_matrix@vectors.T).T
                rotated_vectors = rotated_vectors.reshape(vectors_origin_shape)
                
                data[key][:, :, :2] = rotated_vectors
                
        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != 'flow':
                    data[key] = np.fliplr(data[key]).copy()
                else:
                    data[key][:, :, 0] = -data[key][:, :, 0]

        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != 'flow':
                    data[key] = np.flipud(data[key]).copy()
                else:
                    data[key][:, :, 1] = -data[key][:, :, 1]   
        return data

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
    def __init__(self, ZeroToOne=False):
        super(Normalize, self).__init__()
        self.ZeroToOne = ZeroToOne
        self.num = 0 if ZeroToOne else 0.5

    def __call__(self, data):
        for key in data.keys():
            if key != 'flow':
                data[key] = ((data[key] / 255) - self.num).copy()
        return data

class ToTensor(object):
    def __call__(self, data):
        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()
        return data

class Video_BSD_all_valid_Loader(Dataset):
    def __init__(self, data_path=None, mode="train", crop_size=None, ZeroToOne=False, resizeW=960, resizeH=540, total_frame=5):
        assert data_path, "must have dataset path !"
        self.blur_list = []
        self.sharp_list = []
        self.resizeW = resizeW
        self.resizeH = resizeH
        self.total_frame = total_frame

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])
        
        half_frame = (self.total_frame - 1) // 2
        if data_path:
            for video in sorted(os.listdir(os.path.join(data_path, mode))):
                blur_images_list = sorted(glob.glob(os.path.join(data_path, mode, video, "Blur/RGB", '*.png')))
                for i in range(half_frame, len(blur_images_list) - half_frame):
                    self.blur_list.append(blur_images_list[i - half_frame: i + half_frame + 1])
                self.sharp_list.extend(sorted(glob.glob(os.path.join(data_path, mode, video, "Sharp/RGB", '*.png')))[half_frame:-half_frame])
        
        assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.sharp_list)

    def __getitem__(self, idx):
        sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
        # sharp = cv2.resize(sharp, (self.resizeW, self.resizeH))
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        blurs = []
        for blur_path in self.blur_list[idx]:
            blur = cv2.imread(blur_path).astype(np.float32)
            # blur = cv2.resize(blur, (self.resizeW, self.resizeH))
            blurs.append(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

        sample_tmp = {'sharp': sharp}
        for i in range(self.total_frame):
            sample_tmp[f'blur{i}'] = blurs[i]

        if self.transform:
            sample_tmp = self.transform(sample_tmp)
        
        sample = {'sharp': sample_tmp['sharp'].unsqueeze(0),
                  'blur': torch.cat([sample_tmp[f"blur{i}"].unsqueeze(0) for i in range(self.total_frame)]),
                  'video': self.sharp_list[idx].split('/')[-4],
                  'name': self.sharp_list[idx].split('/')[-1]}

        return sample    

class Video_BSD_json_Loader(Dataset):
    def __init__(self, data_path=None, mode="train", crop_size=None, ZeroToOne=False, resizeW=960, resizeH=540, total_frame=5):
        assert data_path, "must have dataset path !"
        self.blur_list = []
        self.sharp_list = []
        self.crop_region = []
        self.resizeW = resizeW
        self.resizeH = resizeH
        self.total_frame = total_frame

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])

        half_frame = (self.total_frame - 1) // 2

        samples = list()
        with open(data_path, 'r') as file:
            loaded_dict = json.load(file)

            for region in loaded_dict['sharp_regions']:
                
                sharp_img_path = region['path']
                blur_img_path = region['path'].replace('ori','reblur')
                crop_bbox = region['bbox']

                neighbors_list = get_neighbor_imgs_json(blur_img_path)
                
                self.blur_list.append(neighbors_list)
                self.sharp_list.append(sharp_img_path)
                self.crop_region.append(crop_bbox)

        # if data_path:
        #     blur_images_list = sorted(glob.glob(os.path.join(data_path, mode, "blur", '*.png')))
        #     for blur_img in blur_images_list:
        #         neighbors_list = get_neighbor_imgs(blur_img)
        #         self.blur_list.append(neighbors_list)
        #     self.sharp_list.extend(sorted(glob.glob(os.path.join(data_path, mode, "sharp", '*.png'))))
        # assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.sharp_list)

    def __getitem__(self, idx):
        bbox = self.crop_region[idx]
        sharp = cv2.imread(self.sharp_list[idx].replace('4TB','disk2'))
        # sharp = cv2.imread(self.sharp_list[idx].replace("ssddisk","4TB").replace("ID_Blau_results/multiscale_th06_mag5/norm_scale/BSD_1ms8ms","datasets/BSD_1ms8ms/test").replace("ori","Sharp/RGB"))
        sharp = sharp[bbox[0]:bbox[2], bbox[1]:bbox[3]].astype(np.float32)
        # sharp = cv2.resize(sharp, (self.resizeW, self.resizeH))
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        blurs = []
        for blur_path in self.blur_list[idx]:
            blur = cv2.imread(blur_path.replace('4TB','disk2')) # .replace("blurring_output","blurring_output_small/MagAdding20")
            # blur = cv2.imread(blur_path.replace("ssddisk","4TB").replace("ID_Blau_results/multiscale_th06_mag5/norm_scale/BSD_1ms8ms","datasets/BSD_1ms8ms/test").replace("reblur","Blur/RGB").replace("ori","Blur/RGB"))
            blur = blur[bbox[0]:bbox[2], bbox[1]:bbox[3]].astype(np.float32)
            # blur = cv2.resize(blur, (self.resizeW, self.resizeH))
            blurs.append(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

        sample_tmp = {'sharp': sharp}
        for i in range(self.total_frame):
            sample_tmp[f'blur{i}'] = blurs[i]

        if self.transform:
            sample_tmp = self.transform(sample_tmp)
        
        sample = {'sharp': sample_tmp['sharp'].unsqueeze(0),
                  'blur': torch.cat([sample_tmp[f"blur{i}"].unsqueeze(0) for i in range(self.total_frame)])}

        return sample 

def get_image(path):
    transform = transforms.Compose([Normalize(), ToTensor()])
    image = cv2.imread(path).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = {'image': image}
    sample = transform(sample)

    return sample['image']

if __name__ == "__main__":   
    pass
    # dataloader_test = Video_ReblurData_Loader(
    #     generate_path='./dataset/GOPRO_Large_Reblur_Diffusion_M10andO_10000_video',
    #     crop_size=128
    #     )

    # print(dataloader_test.sharp_list[10])
    # print(dataloader_test.blur_list[10])
    # print(dataloader_test[10]['blur'].shape)
    # print(dataloader_test[10]['sharp'].shape)
    # print(len(dataloader_test))
#%%

