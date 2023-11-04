import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple

class DatasetAttack(Dataset):
    def __init__(
        self,
        root:str,
    )->None:
        super().__init__()
        self.root = root
        assert os.path.exists(self.root), "Dataset root is not exists!"

        files = sorted(os.listdir(self.root))
        files = [f for f in files if not f[0]=='.']
        self.files = files
        self.len = len(self.files)
    def _load_image(self, path:str)->np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        filename =  self.files[index]
        path = os.path.join(self.root, filename)
        image = self._load_image(path)
        return filename, image
    def __len__(self)->int:
        return self.len
