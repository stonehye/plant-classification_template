# pip module
import numpy as np
# AI/ML Framework
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
        * description
            - 프로젝트 Dataloader의 기초가 되는 부모 클래스
            - Batch 데이터 생성 및 데이터 shuffling
            - Validation Dataloader 생성
        * inheritance
            - torch.utils.data.DataLoader
        * argument(name : type)
            - dataset : torch.utils.data.Dataset
            - batch_size : int
            - shuffle : boolean
            - validation_split : float(0~1.0)
            - num_workers : int
            - collate_fn : torch.utils.data.dataloader, dafault_collate(default)
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split # set validation split ratio
        self.shuffle = shuffle # set shuffling or not

        self.batch_idx = 0 
        self.n_samples = len(dataset) # 전체 데이터 개수

        # allocate sampler
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # init super class
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        # self.sampler is only for Train data.
        # init super class(DataLoader)
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """
            * description
                - split의 비율에 따라서 train/validation 데이터셋을 나눠주는 sampler
            * argument(name : type)
                - split : float or int
            * return train_sampler, valid_sampler
                - train_sampler : torch.utils.data.sampler.SubsetRandomSampler
                - valid_sampler : torch.utils.data.sampler.SubsetRandomSampler
        """

        # split 값이 유효하지 않을 경우 -> return None, None
        if split == 0.0: 
            return None, None

        idx_full = np.arange(self.n_samples) # sample 개수(n_samples)만큼의 numpy list를 해당 변수에 지정

        # random seed & shuffling  설정
        np.random.seed(0)
        np.random.shuffle(idx_full)

        # (len_valid) == (validation 데이터 개수)
        if isinstance(split, int): # type(split) == int -> 데이터 개수로 데이터 split
            assert split > 0
            assert split < self.n_samples, 'validation set size is configured to be larger than entire dataset.'
            len_valid = split 
        else: # type(split) != int(즉 float일때) -> 비율을 통해서 data split
            len_valid = int(self.n_samples * split)

        # len_valid 변수를 통해서 train/validation split 진행
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # train/validation에 SubsetRandomSampler 할당
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        """
            * description
                - self._split_sampler를 통해서 나눠진 train/validation 데이터에서
                validation 데이터만 사용
                - init시 torch.utils.data.DataLoader를 실행하는 train 데이터와 달리,
                해당 함수를 통해 따로 validation dataloader 따로 실행
            * argument(name : type)
                - using self.valid_sampler, not using argument
            * return torch.utils.data.DataLoader(or None)
        """

        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
