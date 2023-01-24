#built-in
import os
import json
#pip module
import PIL
import glob
#AI/ML Framework
import torch
#custom module
from utils import is_image_file


class TreebeardCustomDataset(torch.utils.data.Dataset):
    """
        * description
            - dataloader아님. 저장된 데이터셋에서 각 파일별로 path를 가져와 주는 역할
                - dataset list(file path list)
            - PIL을 통해서 open image file을 반환 -> __getitem__
            - data 개수 반환 -> __len__
            - 클래스 개수 반환 -> num_classes
        * inheritance
            - torch.utils.data.Dataset
        * argument(name : type)
            - root_dir : str
            - split : str
            - metafile_path : str
            - transforms : torchvision.transforms, None(default)
    """
    def __init__(self, root, split, metafile_path, transforms=None):
        self.root = root # dataset directory 경로
        self.split = split # dataset 하위 폴더 명
        self._dataset_dir =  os.path.join(self.root, self.split)
        
        self.classes = self._get_classes(self._dataset_dir) # class 목록 가져오기
        
        # class(str) : speicies name(str)
        self.class_to_speciesname = self._get_class_to_speciesname(metafile_path)
        # ex) self.class_to_idx = {'13445566' : 'cosmos_bipinnatus', ...}
        
        # class(str) : class_id(int)
        self.class_to_idx = {classname : idx for idx, classname in enumerate(self.classes)}
        # ex) self.class_to_idx = {'dog':0, 'cat':1, 'cow':2, 'bird':3}
        
        # self.data: data(file path) list
        # self.labels: data label list, 입력데이터인 self.data에 1:1로 대응되는 label 데이터
        self.data, self.labels = self._get_data_label()
        # ex) data   = ['data/dog/0.jpg', 'data/dog/1.jpg', 'data/cat/0.jpg', ..., 'data/bird/1.jpg']
        #     labels = [0,0,1,...,3]
        
        self.transforms = transforms # transforms 설정

    def __getitem__(self, idx):
        """
            * description
                - idx에 위치한 img와 label값 반환
            * argument(name : type)
                - idx : int
            * return sample
                - sample : [img, label]
                    - img : PIL.Image.open.conver
                    - lable : int
        """
        img_path, label = self.data[idx], self.labels[idx]
        img = PIL.Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img) # transforms
        
        sample = [img, label]
        return sample

    def __len__(self):
        """
            * description
                - 데이터 개수 반환
        """
        return len(self.data)
    
    def num_classes(self):
        """
            * description
                - 클래스 개수 반환
        """
        return len(self.classes)
    
    def _get_data_label(self):
        """
            * description
                - data, label 리스트 반환
        """
        data, labels = [], []
        for idx, cls in enumerate(self.classes):
            for img in glob.glob(os.path.join(self._dataset_dir, cls, '*')): # class 당 파일
                # 유효한 image file만 사용(이름을 통해서만 필터링, 내부 구성 요소를 통해서도 판단할 수 있어야됨)
                if not is_image_file(img):
                    continue
                data.append(img)
                labels.append(idx)
        return data, labels
    
    def _get_classes(self, dataset_dir):
        """
            * description
                - 클래스 리스트 반환
        """
        classes = os.listdir(dataset_dir) 
        try:
            self.classes.remove('@eaDir')
        except:
            pass
        try:
            self.classes.remove('.DS_Store')
        except:
            pass
        return classes
    
    def _get_class_to_speciesname(self, metafile_path):
        """
            * description
                - 폴더명인 class id와 실제 클래스 영문명이 매칭된 dictionary 반환
            * argument(name : type)
                - metafile_path : str
        """
        class_to_speciesname = None     
        try:
            with open(metafile_path, 'r', encoding='utf-8-sig') as fr: 
                class_to_speciesname = json.load(fr)
        except: 
            if os.environ['RANK'] == '0':
                print('No metafile of dataset.')
        return class_to_speciesname