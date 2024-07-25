from skimage import io, measure
import numpy as np
import os
def isInside(root, video, img_idx):
    video_path = os.path.join(root, video, "Blur/RGB")
    img_list = sorted(os.listdir(video_path))
    if img_idx > (int(img_list[0].split('.')[0])+2) and img_idx < (int(img_list[len(img_list)-1].split('.')[0])-2):
        return True
    # print(video, img_idx)
    return False

def refining_region(regions, binary_image):
    # 想法紀錄：
    # 匡出的region會出現黑色的地方大機率是在邊界，所以檢查四邊的黑色數量是否超過閾值，如果超過就整條砍掉。
    # 依序做左、右、上、下、左、右（左右做兩次是因為上下做完要在refine一次）
    output_bboxs = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        out_minr, out_minc, out_maxr, out_maxc = minr, minc, maxr, maxc

        # left side
        i = out_minc
        while(i < out_maxc):
            left_line = binary_image[out_minr:out_maxr, i:i+1]
            if(np.sum(left_line) / (out_maxr-out_minr)) < 0.5:
                out_minc = i
                i += 1
            else:
                break
        
        # right side
        i = out_maxc
        while(i > out_minc):
            right_line = binary_image[out_minr:out_maxr, i-1:i]
            if(np.sum(right_line) / (out_maxr-out_minr)) < 0.5:
                out_maxc = i
                i -= 1
            else:
                break
        
         # top side
        i = minr
        while(i < maxr):
            top_line = binary_image[i:i+1, out_minc:out_maxc]
            if(np.sum(top_line) / (out_maxc-out_minc)) < 0.5:
                out_minr = i
                i += 1
            else:
                break
        
        # down side
        i = maxr
        while(i > out_minr):
            down_line = binary_image[i-1:i, out_minc:out_maxc]
            if(np.sum(down_line) / (out_maxc-out_minc)) < 0.5:
                out_maxr = i
                i -= 1
            else:
                break

        # left side
        i = out_minc
        while(i < out_maxc):
            left_line = binary_image[out_minr:out_maxr, i:i+1]
            if(np.sum(left_line) / (out_maxr-out_minr)) < 0.5:
                out_minc = i
                i += 1
            else:
                break
        
        # right side
        i = out_maxc
        while(i > out_minc):
            right_line = binary_image[out_minr:out_maxr, i-1:i]
            if(np.sum(right_line) / (out_maxr-out_minr)) < 0.5:
                out_maxc = i
                i -= 1
            else:
                break
        
        # filter by size, both height and width need to be larger than 256
        if((out_maxr-out_minr)>=256)&((out_maxc-out_minc)>=256):
            output_bboxs.append([out_minr, out_minc, out_maxr, out_maxc])       

    return output_bboxs

def get_sharp_region(input_path):
    # Load the image
    image_path = input_path
    image = io.imread(image_path)

    # Check if the image is in the expected boolean format
    if image.ndim == 2 or image.shape[2] == 1:
        # Image is already in boolean format (black & white), where white is regions of interest
        binary_image = image.astype(bool)
    else:
        # If the image is RGB, convert it to boolean (consider white any region that is not pure black)
        binary_image = np.all(image != [0, 0, 0], axis=-1)

    # Label different regions
    label_image = measure.label(binary_image, background=0)
    props = measure.regionprops(label_image)

    # Filter regions based on the size greater than 256x256
    min_size = 256
    filtered_regions = [p for p in props if p.bbox_area > min_size**2]

    return binary_image, filtered_regions

def get_img_path(input_path, video_idx, reblur_result_root, dataset_name):
    img_name = input_path.split('/')[-1]
    out_path = os.path.join(reblur_result_root, dataset_name, video_idx, "ori", img_name)
    return out_path