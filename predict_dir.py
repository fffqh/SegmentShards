import os
import gc
import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

from segment_anything_adv import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

#yaml_path = "./config/base_advCBCL_E1.yaml"
def read_yaml(yaml_path:str):
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()
    return yaml.load(file_data, Loader=yaml.FullLoader)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.45)))


def saveImageWithMask(
    image:np.ndarray,
    anns:List[Any],
    save_path:str,
)->None:
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(anns)
    plt.axis('off')
    plt.savefig(save_path,bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    fig.clf()
    plt.close()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict arg')
    parser.add_argument('--config', default="config/base_advITCVD.yaml", type=str, help='config file')
    args = parser.parse_args()
    yaml_path = args.config
    assert os.path.exists(yaml_path), "config file does not exist!"
    
    params = read_yaml(yaml_path)
    assert os.path.exists(params['predict']['dir_path']), "dir_path does not exist!"
    out_path = params['predict']['out_path']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("create output directory:", out_path)
    
    is_reset = params['predict']['reset']

    dir_path = params['predict']['dir_path']
    files = sorted(os.listdir(dir_path))

    model_type = params['model_type']
    chkpt_path = params['checkpoints'][model_type]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[model_type](checkpoint=chkpt_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=params['points_per_side'],
        points_per_batch=params['points_per_batch'],
        pred_iou_thresh=params['pred_iou_thresh'],
        stability_score_thresh=params['stability_score_thresh'],
        stability_score_offset=params['stability_score_offset'],
        box_nms_thresh=params['box_nms_thresh'],
        crop_n_layers=params['crop_n_layers'],
        crop_nms_thresh=params['crop_nms_thresh'],
        crop_overlap_ratio=params['crop_overlap_ratio'],
        crop_n_points_downscale_factor=params['crop_n_points_downscale_factor'],
        point_grids=params['point_grids'],
        min_mask_region_area=params['min_mask_region_area'],
        output_mode=params['output_mode'],
    )
    
    for file in files:
        if file[0] == '.':
            continue
        if not is_reset:
            if os.path.exists(os.path.join(out_path, file)):
                print('skip:', file)
                continue
        image = cv2.imread(os.path.join(dir_path, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        saveImageWithMask(image, masks, os.path.join(out_path, file))
        print('done:', file)
