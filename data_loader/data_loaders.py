# built-in
import os
# custom module
from base import BaseDataLoader
from .dataset import TreebeardCustomDataset


class TreebeardDataLoader(BaseDataLoader):
    """
        * description
            - Treebeard custom dataset을 학습에 사용할 수 있도록 구성해주는 클래스
        * inheritance
            - base/base_data_loader/BaseDataLoader
        * argument(name : type)
            - augmentation : torchvision.transform
            - root : str
            - split : str
            - data_dir : str
            - metafile_path : str
            - batch_size : int
            - shuffle : boolean, True(default)
            - num_workers : int, 1(default)
            - training : boolean, True(default)
    """
    def __init__(self, augmentation, root, split, metafile_path, batch_size=16, shuffle=True, num_workers=1, training=True):
        self.data_dir = os.path.join(root, split) # data directory 설정
        self.dataset = TreebeardCustomDataset(root, split, metafile_path, transforms=augmentation) # dataset 설정
        
        # init super class(BaseDataLoader)
        super().__init__(self.dataset, batch_size, shuffle, 0., num_workers)
        
