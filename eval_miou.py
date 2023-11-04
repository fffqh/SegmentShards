import os
import csv
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple

from dataset import DatasetCOCOEval, DatasetEval
from segment_anything_adv import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


#yaml_path = "./config/base_advCBCL_E1.yaml"
def read_yaml(yaml_path:str)->Dict:
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()
    return yaml.load(file_data, Loader=yaml.FullLoader)


def mIOU(
    sam_masks:List[np.ndarray],
    ref_masks:List[np.ndarray],
)->float:
    if len(sam_masks) == 0:
        return 0
    assert sam_masks[0].shape == ref_masks[0].shape, "mask shape problem!"
    ious = []
    for s in sam_masks:
        iou_max = 0
        for r in ref_masks:
            inter = np.multiply(s, r) 
            union = np.asarray(s + r > 0 , np.float32) 
            iou = inter.sum() / (union.sum() + 1e-10)
            iou_max = iou if iou > iou_max else iou_max
        ious.append(iou_max)
    return np.mean(ious)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='eval arg')
    parser.add_argument('--config', default="config/base_advITCVD.yaml", type=str, help='config file')
    args = parser.parse_args()
    
    yaml_path = args.config
    params = read_yaml(yaml_path)
    dataset_root = params['dataset_root']
    dataset_adv_root = params['dataset_adv_root']

    assert os.path.exists(dataset_adv_root), "dataset_adv_root does not exist, please check:{}".format(yaml_path)
    assert os.path.exists(dataset_root), "dataset_root does not exist, please check:{}".format(yaml_path)
    
    if params['eval']['save_csv']:
        csv_dir = params['eval']['csv_dir']
        csv_name = 'eval_miou_' + params['eval']['dataset_name'] +'.csv'
        csv_path = os.path.join(csv_dir, csv_name)
        assert not os.path.exists(csv_path), "{} already exists, please delete it.".format(csv_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  

    dataset = DatasetEval(dataset_root,dataset_adv_root)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

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
    
    if params['eval']['save_csv']:
        f = open(csv_path,'w',encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["img_name", "adv_miou"])

    with torch.no_grad():
        for i, data in enumerate(dataloader):

            img_name, imgs, imgs_adv = data

            img_name = img_name[0]
            img = imgs.squeeze(0).numpy()
            img_adv = imgs_adv.squeeze(0).numpy()

            sam_masks = mask_generator.generate(img)        
            adv_masks = mask_generator.generate(img_adv)

            sam_masks_np = [sam_mask["segmentation"].astype(np.uint8) for sam_mask in sam_masks]
            adv_masks_np = [adv_mask["segmentation"].astype(np.uint8) for adv_mask in adv_masks]

            adv_miou = mIOU(sam_masks=adv_masks_np, ref_masks=sam_masks_np)

            if params['eval']['print_miou']:
                print('id:{} img_name:{} adv_miou:{}'.format(i, img_name, adv_miou))
            if params['eval']['save_csv']:
                csv_writer.writerow([img_name, adv_miou])

    if params['eval']['save_csv']:
        f.close()

