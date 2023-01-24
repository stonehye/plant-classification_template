# built-in
import os
import argparse
# pip module
from tqdm import tqdm
from accelerate import Accelerator
# AI/ML Framework
import torch
# custom module
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from config.parse_config import ConfigParser

def main(config):
    """
        * description
            - main test 함수
            - Model, Dataloader, Logger(Wandb, Accelerator), Accelerator(DDP), Metrics 호출
        * argument(name : type)
            - config : OrderedDict(json) object
    """
    accelerator = Accelerator()# Accelerator() 할당
    logger = config.all_logger.get_logger('test')# logger 할당

    # data_loader instance 설정
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # 'config' object에 data class 개수 설정
    config['arch']['args']['num_classes']=data_loader.dataset.num_classes()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # loss, metric 함수 반환
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # accelerator.prepare를 통해서 torch model, dataloader 인스턴스를 accelerator에 사용가능하도록 설정
    model, data_loader = accelerator.prepare(model, data_loader)

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'])

    # accelerator를 통해서 gpu device 설정
    device = accelerator.device
    model = model.to(device)

    # accelerator unwrapping을 통해서 checkpoint loading
    model = accelerator.unwrap_model(model)
    model.load_state_dict(checkpoint['state_dict'], False)
    model.eval()

    # loss 함수 설정
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # test 시작
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            output = model(data)

            #
            # save sample images, or do something with output here
            # 여기다가 샘플 이미지를 저장하든, output을 가지고 뭘 하든 자유롭게 작성해주세요(찡긋)
            #

            # test셋에 대한 loss, metric 계산
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    config_path = '/home/workspace/config/config.json'# 환경변수를 통해서 config.json 파일전달
    config = ConfigParser.from_args(config_path)# ConfigParser 인스턴스를 config 변수에 저장
    main(config)
