import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2
import numpy as np
import pandas as pd
from itertools import islice
import sys
import logging
import tqdm
import argparse
import matplotlib.pyplot as plt
# 將父目錄的路徑添加到 sys.path 中
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from video_dataloader import Video_BSD_all_valid_Loader
from model import ESTRNN
from utils import calc_psnr, same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
import pyiqa
import time
import datetime
from scipy.ndimage import gaussian_filter

def calc_ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32)
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32)
    img2 = img2.transpose((2, 0, 1))

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)


@torch.no_grad()
def valid(model, args, device, title=None):
    model.eval()
    psnr_func = pyiqa.create_metric('psnr', device=device)

    if args.dataset == 'All':
        dataset_name = ['GoPro', 'HIDE']
    else:
        dataset_name = [args.dataset]

    logging.info(f"-----------EVAL------------")
    logging.info(f"Title : {title}")
    results_per_frame = pd.DataFrame(columns=['Dataset', 'Video', 'Image', 'PSNR', 'SSIM', 'Baseline', 'ID-Blau'])
    results_per_video = pd.DataFrame(columns=['Dataset', 'Video', 'PSNR', 'Baseline', 'ID-Blau'])
    previous_video = ''
    total_psnr_per_video = 0
    count_per_video = 0
    for val_dataset_name in dataset_name:
        dataset_path = os.path.join(args.data_path, val_dataset_name)

        # dataset = Video_GoPro_Loader(data_path=dataset_path,
        #                              mode='test',
        #                         crop_size=args.crop_size,
        #                         ZeroToOne=False)
        # dataset = Video_BSD_Loader(data_path=dataset_path,
        #                              mode='test',
        #                         crop_size=args.crop_size,
        #                         ZeroToOne=False)
        dataset = Video_BSD_all_valid_Loader(data_path=dataset_path,
                                     mode='test',
                                crop_size=args.crop_size,
                                ZeroToOne=False)
        dataloader_val = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                drop_last=False)
        
        total_val_psnr = AverageMeter()
        total_val_ssim = AverageMeter()
        tq = tqdm.tqdm(dataloader_val, total=len(dataloader_val))
        tq.set_description(f'Validation {val_dataset_name}')
        start_time = time.time()
        save_dir = os.path.join(args.dir_path, f'{val_dataset_name}') 
        os.makedirs(save_dir, exist_ok=True)
        for idx, sample in enumerate(tq):
            input, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            output = model(input)
            output = output.clamp(-0.5, 0.5)

            psnr = torch.mean(psnr_func(output.detach(), sharp.detach())).item()
            ssim = calc_ssim(output.detach().cpu().numpy().squeeze(), sharp.detach().cpu().numpy().squeeze()).item()
            total_val_psnr.update(psnr)
            total_val_ssim.update(ssim)

            # save_img_path = os.path.join(save_dir, f'{idx:05d}.png')
            save_video_dir = os.path.join(save_dir, sample['video'][0])
            os.makedirs(save_video_dir, exist_ok=True)
            save_img_path = os.path.join(save_video_dir, sample['name'][0])
            save_image(output.squeeze(0).cpu() + 0.5, save_img_path)

            tq.set_postfix(PSNR=total_val_psnr.avg, SSIM=total_val_ssim.avg)

            # new_row = pd.DataFrame({'Dataset': [val_dataset_name], 'Video': [sample['video'][0]],
            #                         'Image': [sample['name'][0]], 'PSNR': [psnr], 'SSIM': [ssim], 
            #                         'Baseline':[args.baseline], 'ID-Blau':['None']})
            # results_per_frame = pd.concat([results_per_frame, new_row], ignore_index=True)

            total_psnr_per_video += psnr
            count_per_video += 1
            if(previous_video != sample['video'][0]) and (previous_video !=''):
                avg_psnr = total_psnr_per_video/count_per_video
                new_row = pd.DataFrame({'Dataset': [val_dataset_name], 'Video': [sample['video'][0]],
                                     'PSNR': [avg_psnr], 'Baseline':[args.baseline], 'ID-Blau':['None']})
                results_per_video = pd.concat([results_per_video, new_row], ignore_index=True)
                total_psnr_per_video = 0
                count_per_video = 0
                
            previous_video = sample['video'][0]

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_obj = datetime.timedelta(seconds=elapsed_time)
        time_str = str(time_obj).split(".")[0]

        logging.info(f"Model : {args.model_path}")
        logging.info(f"Dataset : {val_dataset_name}")
        logging.info(f"The program's running time is (h:m:s) : {time_str}")
        logging.info(f"PSNR : {total_val_psnr.avg:.4f}, SSIM : {total_val_ssim.avg:.4f}\n")
    # results_file = os.path.join(args.dir_path, 'validation_results_per_frame.csv')
    # results_per_frame.to_csv(results_file, index=False)
    # results_file = os.path.join(args.dir_path, 'validation_results_per_video.csv')
    # results_per_video.to_csv(results_file, index=False)

if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='disk2/jthe/datasets', type=str)
    parser.add_argument("--baseline", default='3ms24ms', type=str)
    parser.add_argument("--dir_path", default='home/jthe/DADeblur/DeblurringModel/ESTRNN/ESTRNN_out/BSD_3ms24ms', type=str) ### Need change
    parser.add_argument("--model_path", default='home/jthe/DADeblur/DeblurringModel/ESTRNN/model_weight/BSD_3ms24ms/best_BSD_3ms24ms.pth', type=str)
    parser.add_argument("--title", default='None', type=str)
    parser.add_argument("--dataset", default='BSD_3ms24ms', type=str, choices=['GOPRO_Large']) ### Need change
    parser.add_argument("--crop_size", default=None, type=int)
    # model parameters
    parser.add_argument("--model", default='ESTRNN', type=str, choices=['ESTRNN'])
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    load_model_state = torch.load(args.model_path)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    # Model and optimizer
    net = ESTRNN(args)
    
    #net = nn.DataParallel(net)

    load_model_state = torch.load(args.model_path)
    print("##################################")
    if 'state_dict' in load_model_state.keys():
        load_model_state["state_dict"] = judge_and_remove_module_dict(load_model_state["state_dict"], 'module.model.')
        net.load_state_dict(load_model_state["state_dict"])
        print("1")
    elif 'model_state' in load_model_state.keys():
        net.load_state_dict(load_model_state["model_state"])
        print("2")
    elif 'model' in load_model_state.keys():
        net.load_state_dict(load_model_state["model"])
        print("3")
    elif 'params' in load_model_state.keys():
        net.load_state_dict(load_model_state["params"])
        print("4")
    else:
        net.load_state_dict(load_model_state)
        print("5")
    net.to(device)

    logging.basicConfig(
        filename=os.path.join(args.dir_path, 'eval.log') , format='%(asctime)s | %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    print("device:", device)
    print(f'args: {args}')
    print(f'model: {net}')
    # print(f'model parameters: {count_parameters(net)}')

    same_seed(2023)
    valid(net, args=args, device=device, title=args.title)






# total_psnr = 0.
# total_runtime = 0.
# count = 0
# for _, video in enumerate(os.listdir(data_path)):
#     output_dir = os.path.join(save_dir, video)
#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
#     blur_list = sorted(glob.glob(os.path.join(data_path, video + '/blur', "*")))
#     sharp_list = sorted(glob.glob(os.path.join(data_path, video + '/sharp', "*")))

#     for idx in range(len(blur_list)):
#         count += 1
#         blur = cv2.imread(blur_list[idx]).astype(np.float32) / 255 - 0.5
#         blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
#         blur = torch.from_numpy(blur.transpose((2, 0, 1))).unsqueeze(0).to(device)
#         sharp = cv2.imread(sharp_list[idx]).astype(np.float32) / 255 - 0.5
#         sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
#         sharp = torch.from_numpy(sharp.transpose((2, 0, 1))).unsqueeze(0).to(device)

#         with torch.no_grad():
#             torch.cuda.synchronize()
#             start = time.time()
#             out = net(blur).clamp(-0.5, 0.5)
#             torch.cuda.synchronize()
#             stop = time.time()
#             runtime = stop-start
#             total_runtime += runtime
#             out_name = os.path.split(sharp_list[idx])[-1]
#             total_psnr += calc_psnr(out, sharp)
#             #torchvision.utils.save_image(out + 0.5, os.path.join(output_dir, out_name))

#         print('Video:{}, Frames:{}, Runtime:{:.4f}, Avg Runtime:{:.4f}, Avg PSNR:{:.4f}'.format(video, out_name, runtime,
#                                                                                               total_runtime / count,
#                                                                                               total_psnr / count))





