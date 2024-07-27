import argparse
import os
import torch
import numpy as np
import json
import random
from torch.utils.data import DataLoader
from BME.dataset.dataloader import BlurMagDataset
from BME.model.bme_model import MyNet_Res50_multiscale
from torchvision.transforms.functional import to_pil_image
from utils import *

class RSDM():
    def __init__(self, args) -> None:
        self.args = args
        self.ratio = args.relative_sharpness_ratio
        self.output_setting()
        self.bme_setting()
        self.output_dict = {"sharp_regions": []}

    def output_setting(self):
        self.input_path = os.path.join(self.args.input_dataset_folder, self.args.input_dataset, 'test')
        self.output_json_path = os.path.join(self.args.output_json_folder, self.args.input_dataset+".json")
        self.ouptut_mask_folder = self.args.output_mask_folder
        os.makedirs(self.ouptut_mask_folder, exist_ok=True)
        os.makedirs(self.args.output_json_folder, exist_ok=True)

    def bme_setting(self):
        checkpoint = torch.load(self.args.bme_weight_path, map_location=torch.device('cuda'))
        self.bme = MyNet_Res50_multiscale().cuda()
        self.bme.load_state_dict(checkpoint)

    def run(self):
        video_list = os.listdir(self.input_path)
        for video_c, video in enumerate(video_list, start=1):
            print(f"{video_c} / {len(video_list)}")
            self.run_per_video(video)
        # store the json
        with open(self.output_json_path, 'w') as file:
            json.dump(self.output_dict, file, indent=4)

    def run_per_video(self,video):
        video_path = os.path.join(self.input_path, video)
        output_video_mask_folder = os.path.join(self.ouptut_mask_folder, video)
        os.makedirs(output_video_mask_folder, exist_ok=True)
        temp_list = []
        previous_list = []
        mask_threshold = 0.1
        lower_bound = 0.0
        upper_bound = 1.0
        while True:
            print(f"#########################\nCurrent Video: {video}\nCurrent Mask Threshold: {mask_threshold}")
            
            # Generate Blur Mask
            self.run_bme(video_path, output_video_mask_folder, mask_threshold)

            # Crop Region
            previous_list = temp_list.copy()
            temp_list = []
            mask_list = os.listdir(output_video_mask_folder)
            num_data = int(len(mask_list) * self.ratio)
            print(f"Target num: {num_data}")
            for mask in mask_list:
                mask_path = os.path.join(output_video_mask_folder, mask)
                mask_idx = int(mask.split('.')[0])
                binary_image, regions = get_sharp_region(mask_path)
                refined_bboxs = refining_region(regions, binary_image)
                if refined_bboxs:
                    for bbox in refined_bboxs:
                        if isInside(self.input_path, video, mask_idx):
                            out_path_mag = get_img_path(mask_path, video, self.args.reblur_result_root, self.args.input_dataset)
                            temp_list.append({"path": out_path_mag, "bbox": bbox})

            # Check the number of data
            if abs(len(temp_list) - num_data) < 3:
                print(f"The Number of Sharp Regions: {len(temp_list)}")
                self.output_dict['sharp_regions'].extend(temp_list)
                break
            
            if(mask_threshold < 1e-5):
                print(f"Threshold is too small! Random get {num_data}")
                self.output_dict['sharp_regions'].extend(random.sample(temp_list, num_data))
                break

            if len(temp_list) > num_data:
                upper_bound = mask_threshold
            else:
                lower_bound = mask_threshold
            
            mask_threshold = (upper_bound + lower_bound) / 2

            print(f"Current Number of Sharp Regions: {len(temp_list)}")

    def run_bme(self, dataset_path, output_video_mask_folder, mask_threshold):
        dataset = BlurMagDataset(dataset_root=dataset_path,train=False) 
        test_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        self.bme.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dl):
                img, video_name, file_name = data
                img = img.cuda()
                output = self.bme(img)
                output = output[0].to('cpu').detach().numpy().squeeze()
                mask = output.copy()
                mask[mask >= mask_threshold] = 1
                mask[mask < mask_threshold] = 0
                mask =  (255 - mask * 255).astype(np.uint8)
                mask = to_pil_image(mask)
                output_path = os.path.join(output_video_mask_folder, file_name[0])
                mask.save(output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="BSD_1ms8ms", type=str)
    parser.add_argument("--input_dataset_folder", default="4TB/jthe/datasets/", type=str)
    parser.add_argument("--output_json_folder", default="home/jthe/DADeblur/RSDM/output/json/", type=str)
    parser.add_argument("--output_mask_folder", default="home/jthe/DADeblur/RSDM/output/mask/", type=str)
    parser.add_argument("--reblur_result_root", default="home/jthe/DADeblur/BlurringModel/blurring_output/", type=str)
    parser.add_argument("--bme_weight_path", default="home/jthe/BME/BME/weights/best_net.pth", type=str)
    parser.add_argument("--relative_sharpness_ratio", default=0.2, type=float) # the ratio of relative sharpness region from entire dataset
    args = parser.parse_args()
    rsdm = RSDM(args)
    rsdm.run()