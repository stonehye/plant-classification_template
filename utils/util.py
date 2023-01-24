# built-in
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import json
import yaml
# pip module
import pandas as pd
# AI/ML Framework
import torch



IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def ensure_dir(dirname):
    """
        * description
            - directory 존재여부 확인
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    """
        * description
            - read json
        * return json.load(OrderedDict)
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    """
        * description
            - write json
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def is_image_file(filename):
		return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def inf_loop(data_loader):
    """
        * description
            - wrapper function for endless data loader.
    """
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
        * description
            - 사용가능한 GPU 셋업
            - DataParallel을 위한 gpu device의 index 설정
        * return device, list_ids
            - device : torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            - list_ids : list(range(n_gpu_use))
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    """
        * description
            - Tracking Metric
        * argument(name : type)
            - *keys : tuple
    """
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
