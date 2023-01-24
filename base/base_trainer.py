# built-in
from abc import abstractmethod
import os
# pip module
from numpy import inf
# AI/ML Framework
import torch


class BaseTrainer:
    """
        * description
            - Trainer에 대한 base class
        * argument(name : type)
            - model : torch Model object
            - criterion : torch loss object
            - metric_ftns : metric function
            - optimizer : torch optimizer
            - config : Json object
            - accelerator : Accelerator object
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, accelerator):
        # Set 
        self.config = config
        self.logger = config.all_logger.get_logger('trainer', config['trainer']['verbosity'])
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.accelerator = accelerator

        # train config
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off') # cfg_trainer에 'monitor' key 값이 없으면, 'off' 반환

        # configuration to monitor model performance and save best
        if self.monitor == 'off': # 'monitor' 설정을 하지 않은 경우
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else: # 'monitor' 설정을 한 경우
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            #type(inf) == numpy
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf) # cfg_trainer에 'early_stop' key 값이 없으면 inf 반환
            # self.early_stop 만큼 loss가 연속으로 줄어들지 않는다면, 학습이 종료됨(early stopping)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if self.config['resume'] != '':
            print('resume from ',self.config['resume'])
            self._resume_checkpoint(self.config['resume'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
            * description
                - abstract method
                - 1epoch 실행에 대한 함수
                - 해당 base_trainer를 상속받는 클래스에서 자세하게 코드를 구성
            * argument(name : type)
                - epoch : int
        """
        raise NotImplementedError

    def train(self):
        """
            * description
                - 전체 학습 logic
        """
        not_improved_count = 0 # early stopping을 위한 변수
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # log dict에 log 정보 저장
            train_state = {'epoch': epoch}
            train_state.update(result)
            if os.environ['RANK'] == '0':
                self.config.all_logger.wandb_log(train_state)
            for key, value in train_state.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # metric에 따라서 모델을 평가하고 최고의 checkpoint를 저장
            best = False
            if self.mnt_mode != 'off':
                try:
                    # metric을 통해서 모델 성능이 향상됐는지, 안됐는지 판단
                    improved = (self.mnt_mode == 'min' and train_state[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and train_state[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved: # 성능이 향상됐다면(즉, 지금까지 학습한 결과 중 가장 좋다면)
                    self.mnt_best = train_state[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    if os.environ['RANK'] == '0':
                        self.config.all_logger.wandb_best_alert(epoch, str(result))
                else: # 성능이 향상되지 않았다면
                    not_improved_count += 1

                # 만약에 'self.early_stop'만큼 연속으로 성능향상이 이뤄지지 않으면 학습 중단(Early stopping)
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
            * description
                - checkpoint 저장
                - save_best가 True일 경우 checkpoint 이름을 model_best.pth로 지정
            * argument(name : type)
                - epoch : int
                - save_best : boolean, False(default)
        """
        # Accelerator()를 통해서 학습한 모델의 경우 다른 프로세스(즉 서로 다른 GPU에 할당된 프로세스들)가 모두 끝나길 기다려야됨
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model) # 서로다른 GPU로 부터 model unwrapping
        
        arch = type(self.model).__name__ # 모델 이름 출력
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': unwrapped_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        self.accelerator.save(state, filename) # checkpoint 저장
        self.logger.info('Saving checkpoint: {} ...'.format(filename)) # accelerator.logging
        if save_best: 
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            self.accelerator.save(state, best_path)
            self.logger.info('Saving current best: model_best.pth ...')

    def _resume_checkpoint(self, resume_path):
        """
            * description
                - 저장된 checkpoint를 restore하여 모델에 할당
            * argument(name : type)
                - resume_path : str
        """
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))
        
        checkpoint = torch.load(resume_path) # checkpoint 파일 로딩
        #self.start_epoch = checkpoint['epoch'] + 1 # -> restore했던 때의 모델의 epoch에서부터 restore, 필요없을 것 같아서 삭제함
        self.mnt_best = checkpoint['monitor_best']

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of '
                                'checkpoint. This may yield an exception while state_dict is being loaded.')
        
        # model unwrapping 후(accelerator에서 필수), checkpoint로 부터 architecture parameter 로딩
        unwrap_model = self.accelerator.unwrap_model(self.model)
        unwrap_model.load_state_dict(checkpoint['state_dict'])
        self.model = unwrap_model
        
        # load optimizer state from checkpoint only when optimizer type is not changed.
        # checkpoint로 부터 optimizer state 로딩(optimizer type이 안 변했을시에만)
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. '
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))
