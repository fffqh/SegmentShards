import os
import csv
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from torchmetrics.functional import structural_similarity_index_measure

from dataset import DatasetEval

#yaml_path = "./config/eval_base.yaml"
def read_yaml(yaml_path:str)->Dict:
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()
    return yaml.load(file_data, Loader=yaml.FullLoader)



pixel_std: List[float] = [58.395, 57.12, 57.375]

if __name__ == '__main__':
    pixel_mean: List[float] = [123.675, 116.28, 103.53]
    pixel_std: List[float] = [58.395, 57.12, 57.375]
    
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    parser = argparse.ArgumentParser(description='eval arg')
    parser.add_argument('--config', default="config/base_advITCVD.yaml", type=str, help='config file')
    args = parser.parse_args()
    yaml_path = args.config
    assert os.path.exists(yaml_path), "config file does not exist"
    
    
    params = read_yaml(yaml_path)
    dataset_root = params['dataset_root']
    dataset_adv_root = params['dataset_adv_root']

    assert os.path.exists(dataset_adv_root), "dataset_adv_root does not exist, please check:{}".format(yaml_path)
    assert os.path.exists(dataset_root), "dataset_root does not exist, please check:{}".format(yaml_path)
   
    if params['eval']['save_csv']:
        csv_dir = params['eval']['csv_dir']
        csv_name = 'eval_ssim_' + params['eval']['dataset_name'] +'.csv'
        csv_path = os.path.join(csv_dir, csv_name)
        assert not os.path.exists(csv_path), "{}already exists, please delete it.".format(csv_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

    # 加载数据集
    dataset = DatasetEval(dataset_root,dataset_adv_root)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    
    if params['eval']['save_csv']:
        #准备csv文件
        f = open(csv_path, 'w',encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["img_name", "adv_ssim"])
    
    
    for i, data in enumerate(dataloader):
        
        img_name, imgs, imgs_adv = data
        
        img_name = img_name[0]
        imgs = imgs.permute(0,3,1,2).float()
        imgs_adv = imgs_adv.permute(0,3,1,2).float()
        
        ssim = structural_similarity_index_measure(imgs_adv, imgs).numpy()
        if params['eval']['print_ssim']:
            print('id:{} img_name:{} adv_ssim:{}'.format(i, img_name, ssim))
        if params['eval']['save_csv']:
            csv_writer.writerow([img_name, ssim])

    if params['eval']['save_csv']:
        f.close()



