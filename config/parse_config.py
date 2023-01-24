# built-in
from datetime import datetime
import logging
import os
from pathlib import Path
from functools import partial
# custom module
from logger import Logger
from utils import read_json, write_json


class ConfigParser:
    """
        * description
            - configuration 파일(json)을 parsing하기 위한 클래스
            - 학습을 위한 hyperparameter, module 초기화, checkpoin 저장 등을 다루는데 사용됨
        * argument(name : type)
            - config : OrderedDict(from json)
                - config.json 파일로부터 읽어온 json타입을 OrderedDict 타입으로 변환하여 전달됨
            - run id : datetime, None(default)
                - checkpoint나 log 저장할 때 사용
    """
    def __init__(self, config, run_id=None):
        self._config = config

        # trained model, log를 저장할 path 설정
        save_dir = Path(self.config['trainer']['save_dir'])

        # config['name']을 architecture이름으로 가져갈지, 아니면 따로 지정할 지 고민 필요
        exper_name = self.config['name']
        if run_id is None: # time-stamp를 기본 run_id로 사용
            run_id = datetime.now().strftime(r'%y%m%d_%H%M')
        self._save_dir = save_dir / 'models' / exper_name / run_id # model file 저장 path
        self._log_dir = save_dir / 'log' / exper_name / run_id # log file 저장 path

        # checkpoint, log 저장할 directory 생성(mkdir)
        # @property, def save_dir(): return self._save_dir
        # self.save_dir.mkdir == self._save_dir.mkdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 업데이트된 config 파일을 checkpoint 저장하는 path에 저장
        write_json(self.config, self.save_dir / 'config.json')

        # logging module 설정
        self.all_logger = Logger(self.log_dir, os.environ['RANK'], self.config['logger'])
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls,config_path):
        """
            * description
                - @classmethod
                - cli argument로 부터 class initialization, train, test에서 사용
                - Class의 객체 선언없이도 접근가능한 함수
            * argument(name : type)
                - cls : 해당 클래스 그 자체를 의미(내가 헷갈려서..)
                - config_path : config(json) directory
            * return cls
                - cls : ConfigParser itself
        """
        config = read_json(config_path)
        return cls(config) # cls == ConfigParser


    def init_obj(self, name, module, *args, **kwargs):
        """
            * description
                - config에서 'type'을 통해 지정된 module의 이름을 찾고 해당 module로 초기화된 instance 반환
                - `object = config.init_obj('name', module, a, b=1)`
                  ==
                  `object = module.name(a, b=1)`
            * argument(name : type)
                - name : str
                - module : Class
                - *args : tuple
                - **kwargs : dictionary
            * return cls
                - cls : ConfigParser itself
        """
        # 'self[name]' == 'self.config[name]'
        # 'type(self.config)' == 'json Object'
        module_name = self[name]['type']
        module_args = dict(self[name]['args']) # module arguement(in json file)
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
            * description
                -  Finds a function handle with the name given as 'type' in config 
                   and returns the function with given arguments fixed with functools.partial.
                -  config에서 'type'을 통해 지정된 function 이름을 찾고 functools.partial과 주어진 argument를 통해서 함수를 반환
                - `function = config.init_ftn('name', module, a, b=1)`
                   ==
                  `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
            * argument(name : type)
                - name : str
                - module : Some Class
                - *args : tuple
                - **kwargs : dictionary

            * return cls
                - cls : ConfigParser itself
        """
        # 'self[name]' == 'self.config[name]'
        # 'type(self.config)' == 'json Object'
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """
            * description
                - 해당 함수를 통해서 Class를 Dictionary 처럼 사용하여 데이터에 접근할 수 있게 해줌
                - 일반 dictionary 처럼 item(self.config)에 접근
            * argument(name : type)
                - name : str
            * return self.config[name]
                - 'self.config' is Json Object
        """
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir