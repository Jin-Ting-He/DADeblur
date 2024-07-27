import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms 
from ID_Blau.flow_viz import flow_to_image
from ID_Blau.utils import same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
from ID_Blau.diffusion_network import DDIM, DDPM
from ID_Blau.diffusion_model import UNet
from ID_Blau.utils_.utils import *

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
    
class BlurringModel():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        same_seed(self.args.seed)
        load_model_state = torch.load(self.args.model_path)
        model_args = load_model_state['args']

        beta = self.generate_linear_schedule(
            model_args.num_timesteps, model_args.beta_1, model_args.beta_T)
        
        model_UNet = UNet(
            channel_mults=model_args.channel_mults,
            base_channels=model_args.base_channels,
            time_dim=model_args.time_dim,
            dropout=model_args.dropout
            ).to(self.device)
        
        if args.model == "DDIM":
            self.diffusionModel = DDIM(model_UNet, img_channels=9, betas=beta).to(self.device)
        elif args.model == "DDPM":
            self.diffusionModel = DDPM(model_UNet, img_channels=9, betas=beta).to(self.device)
        else:
            raise ValueError(f"model not supported {args.model}")
        
        if 'model_state' in load_model_state.keys():
            self.diffusionModel.load_state_dict(load_model_state["model_state"])
        else:
            self.diffusionModel.load_state_dict(load_model_state)
        
        self.dir_list = list_directories(self.args.input_folder_path)

    def get_dataset_len(self):
        self.dataset_len = 0
        self.cur = 0
        for dir in self.dir_list:
            cur_video_frames_folder = dir+ "/Sharp/RGB"
            image_path_list = list_files_sorted(cur_video_frames_folder)
            for image in image_path_list:
                self.dataset_len+=1

    def generate_new_training_data(self):
        self.get_dataset_len()
        for dir in self.dir_list:
            cur_video_frames_folder = dir+ "/Blur/RGB"
            video_idx = dir.split('/')[-1]
            cur_video_flow_folder = self.args.blur_condition_folder_path + video_idx
            output_video_folder = self.args.output_folder_path + video_idx
            os.makedirs(output_video_folder, exist_ok=True)
            
            output_ori_frames_folder = os.path.join(output_video_folder, "ori")
            os.makedirs(output_ori_frames_folder, exist_ok=True)
            output_reblur_frames_folder = os.path.join(output_video_folder, "reblur")
            os.makedirs(output_reblur_frames_folder, exist_ok=True)
            output_flow_folder = os.path.join(output_video_folder,"flow")
            os.makedirs(output_flow_folder, exist_ok=True)

            image_path_list = list_files_sorted(cur_video_frames_folder)

            for image in image_path_list:
                print(f"{self.cur} / {self.dataset_len}")
                self.cur+=1
                image_name = image.split('/')[-1]
                flow_name = image_name.replace('.png','.npy')
                flow = cur_video_flow_folder+'/'+flow_name
                if os.path.exists(flow) and os.path.exists(image):
                    # print(image,flow)
                    flow, reblur_image = self.blurring(image, flow, sample_timesteps=self.args.sample_timesteps)

                    ori_image = cv2.imread(image)

                    ori_img_path = os.path.join(output_ori_frames_folder,image_name)
                    cv2.imwrite(ori_img_path, ori_image)
                    reblur_img_path = os.path.join(output_reblur_frames_folder,image_name)
                    cv2.imwrite(reblur_img_path, reblur_image)
                    flow_img_path = os.path.join(output_flow_folder,image_name)
                    cv2.imwrite(flow_img_path, flow)

    def blurring(self, image_path, flow_path, sample_timesteps):
        """use dataset to val and save image"""

        with torch.no_grad():
            self.diffusionModel.eval()
            transform = transforms.Compose([Normalize(), ToTensor()])
            sharp = cv2.imread(image_path).astype(np.float32)
            sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
            flow = np.load(flow_path)
            flow = flow.astype(np.float32)

            # magnitude_max=np.max(flow[2])

            flow[2] = flow[2]/150
            flow[2][flow[2]>1] = 1
            flow = flow.transpose((1, 2, 0))

            sample = {'sharp': sharp,
                    'flow': flow}

            sample = transform(sample)

            sharp = sample['sharp'].unsqueeze(0).to(self.device)
            flow = sample['flow'].unsqueeze(0).to(self.device)

            condition = torch.cat([sharp, flow], dim=1)
            if args.model == "DDIM":
                output = self.diffusionModel.sample(condition=condition, sample_timesteps=sample_timesteps, device=self.device, tqdm_visible=True)
            elif args.model == "DDPM":
                output = self.diffusionModel.sample(condition=condition, device=self.device, tqdm_visible=True)
            output = output.clamp(-0.5, 0.5)

            #---------flow--------------------
            flow = flow.squeeze(0).cpu().numpy()
            flow = flow.transpose((1,2,0))
            flow_x = flow[:, :, 0] * flow[:, :, 2]
            flow_y = flow[:, :, 1] * flow[:, :, 2]
            optical_flow = np.stack((flow_x, flow_y), axis=-1)

            # print("optical_flow", optical_flow[250][250])
            flo = flow_to_image(optical_flow, norm=1)

            return flo[:, :, [2,1,0]], tensor2cv(output + 0.5)
        
    def generate_linear_schedule(self, T, beta_1, beta_T):
        return torch.linspace(beta_1, beta_T, T).double()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="home/jthe/Deblur_Domain_Adaptation/data_generator/data_reblur/model_weights/epoch_5000_diffusion_flow_reblur.pth", type=str)
    parser.add_argument("--input_folder_path", default='4TB/jthe/datasets/BSD_3ms24ms/test/', type=str)
    parser.add_argument("--blur_condition_folder_path", default="home/jthe/DADeblur/DBCGM/output/BSD_3ms24ms/test/", type=str)
    parser.add_argument("--output_folder_path", default="home/jthe/DADeblur/BlurringModel/blurring_output/BSD_3ms24ms/test/", type=str)
    parser.add_argument("--model", default='DDIM', type=str)
    parser.add_argument("--sample_timesteps", default=20, type=int)
    parser.add_argument("--seed", default=2023, type=int)

    args = parser.parse_args()

    blurring_model = BlurringModel(args)
    blurring_model.generate_new_training_data()