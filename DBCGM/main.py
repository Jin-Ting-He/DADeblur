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

class BME():
    def __init__(self, args) -> None:
        self.args = args

    def inference(self):
        # load model
        checkpoint_path = os.path.join(self.args.weight_path, self.args.model_name)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        self.model = MyNet_Res50_multiscale().cuda()
        self.model.load_state_dict(checkpoint)
        # print("Loading Model Done")

        # load dataset
        infer_dataset = BlurMagDataset(dataset_root=self.args.infer_dataset_path,train=False)
        infer_dl = DataLoader(infer_dataset, batch_size=1, shuffle=False, pin_memory=True)
        # print("Loading Dataset Done")

        output_folder = self.args.infer_output_path
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # argument for BME
    parser.add_argument("--weight_path", default="home/jthe/BME/BME/weights/", type=str)
    parser.add_argument("--infer_dataset_path", default="4TB/jthe/datasets/BSD_2ms16ms/test", type=str)
    parser.add_argument("--infer_output_path", default="home/jthe/DADeblur/DBCGM/BME/output/BSD_2ms16ms/", type=str)
    parser.add_argument("--model_name", default="best_net.pth", type=str)
    args = parser.parse_args()

    # blur magnitude estimation
    bme = BME(args)
    bme.inference()