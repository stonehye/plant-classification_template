# built-in
import os
# pip module
from accelerate import Accelerator
import numpy as np
# AI/ML Framework
import torch
import torchvision.transforms as transforms
#custom module
from config.parse_config import ConfigParser
from trainer import Trainer
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch


def main(config):
    """
        * description
            - main train 함수
            - Trainer 클래스 호출
            - Model, Dataloader, Logger(Wandb, Accelerator), Accelerator(DDP), Loss, Metrics, Optimizer, LR Scheduler 설정
        * argument(name : type)
            - config : OrderedDict(json) object
    """
    # fix random seeds for reproducibility
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED) # torch seed 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    accelerator = Accelerator() # Accelerator() 할당
    logger = config.all_logger.get_logger('train') # logger 할당

    # augmentation 세팅
    aug_list = []
    for k, v in config['augmentation'].items(): # config 파일로 부터 요소 하나씩 init하여 list에 저장
        if k == 'RandomApply': # Transform이 RandomApply인 경우 파라미터를 따로 지정
            random_aug_list = [getattr(transforms, rk)(**rv) 
                               for rk, rv in config['augmentation'][k]['transforms'].items()] 
            prob = config['augmentation'][k]['p']
            aug_list.append(getattr(transforms, k)(transforms=random_aug_list, p=prob))
        else:
            aug_list.append(getattr(transforms, k)(**v))
    augmentation = transforms.Compose(aug_list)

    # 'config' object와 augmentation 변수를 통해서 train/validation Dataloader 호출
    data_loader = config.init_obj('data_loader', module_data, augmentation)
    valid_data_loader = config.init_obj('valid_data_loader', module_data, augmentation)
    
    # 'config' object에 data class 개수 설정
    config['arch']['args']['num_classes']=data_loader.dataset.num_classes()
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # loss, metric 함수 반환
    criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics = [getattr(module_metric, met['type'])(**dict(met['args'])) for met in config['metrics']] # TODO: Need to be confirmed
    
    if os.environ['RANK'] == '0':
        config.all_logger.wandb_watch(model, criterion) # model에 대한 wandb watch 설정

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # accelerator.prepare를 통해서 torch model, dataloader, optimizer, lr_scheduler  인스턴스를 accelerator에 사용가능하도록 설정
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
          model, optimizer, data_loader, lr_scheduler
    )
    
    # accelerator를 통해서 gpu device 설정
    device = accelerator.device
    model = model.to(device) # model을 accelerator를 통해서 gpu device에 할당
    
    valid_data_loader = accelerator.prepare(valid_data_loader) # convert validation dataloader object into Accelerator object

    # Trainer obejct선언
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      accelerator=accelerator,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    # training 시작
    trainer.train()
    
if __name__ == '__main__':
    config_path = os.environ['CONFIG_PATH'] # 환경변수에 저장된 config.json 파일경로 전달
    config = ConfigParser.from_args(config_path) # ConfigParser 인스턴스를 config 변수에 저장
    main(config)