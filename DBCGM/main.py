import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import cv2
import random
import argparse
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# set random seed
torch.manual_seed(39)
torch.cuda.manual_seed(39)
random.seed(39)
np.random.seed(39)
from torch.utils.data import DataLoader
from BME.dataset.dataloader import BlurMagDataset
from BME.model.bme_model import MyNet_Res50_multiscale

from BOE.core.raft import RAFT
from BOE.core.utils import flow_viz
from BOE.core.utils.utils import *

class BME(): # Blur Magnitude Estimator
    def __init__(self, args) -> None:
        self.args = args

    def inference(self):
        # load model
        checkpoint_path = os.path.join(self.args.bme_weight_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        self.model = MyNet_Res50_multiscale().cuda()
        self.model.load_state_dict(checkpoint)
        # print("Loading Model Done")

        # load dataset
        infer_dataset = BlurMagDataset(dataset_root=self.args.infer_dataset_path,train=False)
        infer_dl = DataLoader(infer_dataset, batch_size=1, shuffle=False, pin_memory=True)
        # print("Loading Dataset Done")

        output_folder = self.args.bme_output_path
        os.makedirs(output_folder, exist_ok=True)

        # print("Starting Evaluation")
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(infer_dl, total=len(infer_dl))
            pbar.set_description(f'Generating Blur Magnitude')
            for i,data in enumerate(infer_dl):
                # print("Complete: ", i)
                blur_img, video_name, file_name = data
                blur_img = blur_img.cuda()

                output = self.model(blur_img)
                # output = output.clamp(-0.5, 0.5)
                output = output.clamp(0 ,1)
                output = output[0].to('cpu').detach().numpy().squeeze()
                # output =  ((output+0.5) * 205)
                output =  ((output) * 205)
                # output = output/np.max(output)
                # output = np.uint8(255-(output))

                output_video_folder = os.path.join(output_folder, video_name[0])
                os.makedirs(output_video_folder, exist_ok=True)
                output_path = os.path.join(output_video_folder,file_name[0].replace(".png",".npy"))
                # cv2.imwrite(output_path, output)
                np.save(output_path, output)
                pbar.update(1)
        pbar.close()    

class BOE(): # Blur Orientation Estimator
    def __init__(self, args):
        self.args = args
        self.dir_list = list_directories(self.args.infer_dataset_path)
        self.list_of_filelist = [list_files_sorted(dir+"/Blur/RGB") for dir in self.dir_list]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.boe_weight_path))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        self.c = 0
    
    def infernece(self):
        pbar = tqdm(self.list_of_filelist, total=len(self.list_of_filelist))
        pbar.set_description(f'Generating Blur Orientation')
        for filelist in self.list_of_filelist:
            self.c+=1
            self.generate(filelist)
            pbar.update(1)
        pbar.close() 

    def generate(self, frames_file_list):
        magnitude_max=torch.tensor([0]).cuda()
        video_idx = frames_file_list[0].split('/')[-4]
        flow_video_path = os.path.join(self.args.boe_output_path, video_idx)
        os.makedirs(flow_video_path, exist_ok=True)
        flow_viz_path =  flow_video_path.replace('test','viz')
        os.makedirs(flow_viz_path, exist_ok=True)
        with torch.no_grad():
            for idx in range(1,len(frames_file_list)-1):
                # print("complete: ",idx)
                flow0 = 0
                flow1 = 0

                ###### forward direciton ##########
                origin_image1_path = frames_file_list[idx]
                origin_image2_path = frames_file_list[idx+1]
                image1 = self.load_image(origin_image1_path)
                imagen = self.load_image(origin_image2_path)

                padder = InputPadder(image1.shape)
                image1, imagen = padder.pad(image1, imagen)

                flow_low_1n, flow_up_1n = self.model(image1, imagen, iters=args.epochs, test_mode=True)
                
                flow0 = flow_up_1n

                ####### reverse direction ###########
                origin_image1_path = frames_file_list[idx]
                origin_image2_path = frames_file_list[idx-1]
                image1 = self.load_image(origin_image1_path)
                imagen = self.load_image(origin_image2_path)
        
                padder = InputPadder(image1.shape)
                image1, imagen = padder.pad(image1, imagen)
        
                flow_low_1n, flow_up_1n = self.model(image1, imagen, iters=args.epochs, test_mode=True)
                
                flow1 -= flow_up_1n


                flow0 = flow0.cpu().squeeze().numpy()
                flow1 = flow1.cpu().squeeze().numpy()

                flow = (flow0 + flow1) / 2
                magnitude = np.linalg.norm(flow, axis=0)
                magnitude = np.expand_dims(magnitude, axis=0)

                flow = flow / magnitude #transfer to unit vector
                info = np.concatenate((flow, magnitude), axis=0).astype(np.float32)

                save_name = frames_file_list[idx].split("/")[-1].replace(".png", ".npy")
                save_path = os.path.join(flow_video_path, save_name)
                np.save(save_path, info)
                viz_save_name = frames_file_list[idx].split("/")[-1]
                viz_save_path = os.path.join(flow_viz_path, viz_save_name)
                self.viz(info, viz_save_path)

                if np.max(magnitude) > magnitude_max:
                    magnitude_max=np.max(magnitude)

                        #viz(blurry_image, temp)
        # print("magnitude_max", magnitude_max)

    def load_image(self, imfile, device='cuda'):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(device)

    def viz(self, flow, save_path):
        flow = flow.astype(np.float32)
        magnitude_max=np.max(flow[2])
        flow[2] = flow[2]/magnitude_max
        flow = flow.transpose((1, 2, 0))
        flow_x = flow[:, :, 0] * flow[:, :, 2]
        flow_y = flow[:, :, 1] * flow[:, :, 2]
        optical_flow = np.stack((flow_x, flow_y), axis=-1)
        flo = flow_viz.flow_to_image(optical_flow)
        flo = flo[:, :, [2,1,0]]
        cv2.imwrite(save_path, flo)

class DBCGM(): # Domain-adaptive Blur Condition Generation Module
    def __init__(self, args):
        self.args = args
        self.magnitude_folder = self.args.bme_output_path
        self.orientation_folder = self.args.boe_output_path
        self.output_folder = self.args.dbcgm_output_path
        self.video_list = sorted(os.listdir(self.orientation_folder))

    def generate_dbc(self):
        pbar = tqdm(self.video_list, total=len(self.video_list))
        pbar.set_description(f'Generating Domain-adaptive Blur Condition')
        for video in self.video_list:
            file_list = sorted(os.listdir(os.path.join(self.orientation_folder, video)))
            outptu_video_path = os.path.join(self.output_folder, video)
            os.makedirs(outptu_video_path, exist_ok=True)
            for i in range(len(file_list)):
                # Orientation
                file = file_list[i]
                file_path = os.path.join(self.orientation_folder, video, file)
                blur_condition = np.load(file_path)

                # Magnitude
                cur_blur_mag_path = os.path.join(self.magnitude_folder, video, file)
                cur_blur_mag = np.load(cur_blur_mag_path)

                if(i>=2)&(i<(len(file_list)-2)):
                    # neighbors blur mask
                    t_minus2_path = os.path.join(self.magnitude_folder,video, file_list[i-2])
                    t_minus2_mask = np.load(t_minus2_path)
                    t_minus1_path = os.path.join(self.magnitude_folder,video, file_list[i-1])
                    t_minus1_mask = np.load(t_minus1_path)
                    t_plus2_path = os.path.join(self.magnitude_folder,video, file_list[i+2])
                    t_plus2_mask = np.load(t_plus2_path)
                    t_plus1_path = os.path.join(self.magnitude_folder,video, file_list[i+1])
                    t_plus1_mask = np.load(t_plus1_path)
                    neighbor_avg_magnitude = (t_minus2_mask + t_minus1_mask + t_plus2_mask + t_plus1_mask)/4
                    cur_blur_mag = cur_blur_mag / np.max(cur_blur_mag)
                    new_blur_magnitude = cur_blur_mag*neighbor_avg_magnitude
                    blur_condition[2] = new_blur_magnitude
                else:
                    blur_condition[2] = cur_blur_mag.copy()

                output_file_path = os.path.join(outptu_video_path, file)
                np.save(output_file_path, blur_condition)
            pbar.update(1)
        pbar.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # argument for BME
    parser.add_argument("--bme_weight_path", default="home/jthe/BME/BME/weights/best_net.pth", type=str)
    parser.add_argument("--infer_dataset_path", default="4TB/jthe/datasets/BSD_2ms16ms/test", type=str)
    parser.add_argument("--bme_output_path", default="home/jthe/DADeblur/DBCGM/BME/output/BSD_2ms16ms/test", type=str)
    # argument for BOE
    parser.add_argument("--boe_weight_path", default="home/jthe/Deblur_Domain_Adaptation/data_generator/blur_orientation_estimator/weights/raft-things.pth", type=str)
    parser.add_argument("--boe_output_path", default="home/jthe/DADeblur/DBCGM/BOE/output/BSD_2ms16ms/test", type=str)
    parser.add_argument('--epochs',default=20, help="iter times")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # argument for DBCGM
    parser.add_argument("--dbcgm_output_path", default="home/jthe/DADeblur/DBCGM/output/BSD_2ms16ms/test", type=str)
    args = parser.parse_args()

    # blur magnitude estimation
    bme = BME(args)
    bme.inference()

    # blur orientation estimation
    boe = BOE(args)
    boe.infernece()

    # domain-adaptive blur condition generation
    dbcgm = DBCGM(args)
    dbcgm.generate_dbc()