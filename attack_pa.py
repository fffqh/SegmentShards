import os
import gc
import cv2
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from torchvision.transforms.functional import to_pil_image 

from dataset import DatasetCOCO, DatasetAttack
from segment_anything_adv import SamTargetAdversarialGenerator,sam_model_registry

#yaml_path = "./config/base_advCBCL_E1.yaml"
def read_yaml(yaml_path:str)->Dict:
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
        ax.imshow(np.dstack((img, m*0.35)))

def saveImageWithMask(
    image:np.ndarray,
    anns:List[Any],
    save_path:str,
)->None:
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(anns)
    plt.axis('off')
    plt.savefig(save_path)
    fig.clf()
    plt.close()
    gc.collect()


def _process_cocodata(
    data:Tuple[Any, Any, Any]
)->Tuple[np.ndarray, str]:
    ids, imgs, _ = data
    imgid = ids[0].numpy()
    filename = '{:0>12d}.jpg'.format(imgid)
    img = imgs.squeeze(0).numpy()
    return img, filename

def _process_attackdata(
    data:Tuple[Any, Any],
)->Tuple[np.ndarray, str]:
    filenames, imgs = data
    filename = filenames[0]
    img = imgs.squeeze(0).numpy()
    return img, filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack arg')
    parser.add_argument('--config', default="config/base_advITCVD.yaml", type=str, help='config file')
    args = parser.parse_args()
    yaml_path = args.config
    assert os.path.exists(yaml_path), "config file does not exist!"
    
    params = read_yaml(yaml_path)
    tg_path = params['attack']['target_image_path']
    advout_path = params['attack']['advout_path']
    advout_name = params['attack']['attack_name']
    e,eps,iter = str(params['adv_e']),\
                 str(params['adv_epsilon']),\
                 str(params['adv_iter'])
    point_per_side, point_per_batch, stb = str(params['points_per_side']),\
                                           str(params['points_per_batch']),\
                                           str(params['stability_score_thresh'])
    advout_path_params = advout_path + '_' + e + '_' + eps +'_' + iter
    advout_path_params += '_' + point_per_side + '_' + point_per_batch + '_' + stb
    advout_path_params_name = os.path.join(advout_path_params, advout_name)

    assert os.path.exists(tg_path), "target image is not exists!"
    if not os.path.exists(advout_path_params_name):
        os.makedirs(advout_path_params_name)
        print("create output directory:", advout_path_params_name)
    
    dataset_type = params['dataset_type']
    if dataset_type == 'coco':
        dataset = DatasetCOCO(params['dataset_root'], params['annfile_path'])
    elif dataset_type == 'adv':
        dataset = DatasetAttack(params['dataset_root'])
    else:
        assert True, 'dataset type error!'
    
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    model_type = params['model_type']
    chkpt_path = params['checkpoints'][model_type]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[model_type](checkpoint=chkpt_path)
    sam.to(device=device)
    adv_generator = SamTargetAdversarialGenerator(
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
                        return_logists=params['return_logists'],
                        adv_e=params['adv_e'],
                        adv_epsilon=params['adv_epsilon'],
                        adv_iter=params['adv_iter']
                    )

    tg_img = cv2.imread(tg_path)
    tg_img = cv2.cvtColor(tg_img, cv2.COLOR_BGR2RGB)
    N = len(dataloader)
    for i, data in enumerate(dataloader):
        if i >= params['attack']['image_number']:
            break
        
        if dataset_type == 'coco':
            img, filename = _process_cocodata(data)
        elif dataset_type == 'adv':
            img, filename = _process_attackdata(data)

        if os.path.exists(os.path.join(advout_path_params_name, filename)):
            print('attack finish:{}/{}'.format(i,N))
            continue

        adv_masks, adv_img = adv_generator.run(img, tg_img)

        to_pil_image(adv_img).save(
            os.path.join(advout_path_params_name, filename))
        print('attack finish:{}/{}'.format(i, N))

