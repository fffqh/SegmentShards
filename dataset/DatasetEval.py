import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple

class DatasetEval(Dataset):
    def __init__(
        self,
        root:str,
        rootadv:str,
    )->None:
        super().__init__()
        self.root = root
        self.rootadv = rootadv
        assert os.path.exists(self.root), "Dataset root is not exists!"
        assert os.path.exists(self.rootadv), "Dataset rootadv is not exists!"

        files = sorted(os.listdir(self.rootadv))
        files = [f for f in files if not f[0]=='.']
        self.files = files
        self.len = len(self.files)

    def _load_image(self, path:str)->np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def __getitem__(self, index:int) -> Tuple[Any, Any, Any]:
        filename =  self.files[index]
        path = os.path.join(self.root, filename)
        pathadv = os.path.join(self.rootadv, filename)

        image = self._load_image(path)
        image_adv = self._load_image(pathadv)
        return filename, image, image_adv
    
    def __len__(self)->int:
        return self.len
