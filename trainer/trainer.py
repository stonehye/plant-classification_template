import os
# AI/ML Framework
import torch
# custom module
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
        * description
            - Trainer 클래스(train.py에서 호출됨)
        * inheritance
            - BaseTrainer
        * argument(name : type)
            - model : torch Model object
            - criterion : torch loss object
            - metric_ftns : metric function
            - optimizer : torch optimizer
            - config : Json object
            - accelerator : Accelerator object
            - device : str(accelerator.device)
            - data_loader : torch.utils.data.DataLoader
            - valid_data_loader : torch.utils.data.DataLoader, None(default)
            - lr_scheduler : torch.optim.lr_scheduler, None(default)
            - len_epoch : int, None
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, accelerator, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, accelerator)
        self.device = device
        self.data_loader = data_loader
        
        if len_epoch is None:
            # epoch 기반 학습
            self.len_epoch = len(self.data_loader) # == dataest_size / batch_size
        else:
            # iteration 기반 학습
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config['log_step']

        self.train_metrics = MetricTracker('loss', *[m.key_name for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.key_name for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
            * description
                - BaseTrainer._train_epoch으로 부터 상속받은 method
                - 1 epoch 진행
            * argument(name : type)
                - epoch : int
            * return epoch_state
                - epoch_state = self.train_metrics.result()
        """
        # train 세팅
        self.model.train()
        self.train_metrics.reset()

        # start training (1epoch)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            self.optimizer.zero_grad() # set graident 0
            output = self.model(data) # model prediction(output)
            loss = self.criterion(output, target)
            
            if os.environ['RANK'] == '0':
                self.config.all_logger.wandb_log({'train_loss(iter)':loss}, step=self.len_epoch*(epoch-1)+batch_idx)
            self.accelerator.backward(loss) # backward using Accelerator(DDP)

            self.optimizer.step()
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.key_name, met.measure(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item())
                )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            #if batch_idx == self.len_epoch:
            #    break
            
        if os.environ['RANK'] == '0':
            self.config.all_logger.wandb_log({'train_loss(epoch)':loss}, step=epoch)
        epoch_state = self.train_metrics.result()
        
        if self.do_validation: # if validation is True 
            val_state = self._valid_epoch(epoch)
            epoch_state.update(**{'val_'+k : v for k, v in val_state.items()})

        return epoch_state

    def _valid_epoch(self, epoch):
        """
            * description
                - 1 epoch 진행 후 validation 진행
            * argument(name : type)
                - epoch : int
            * return self.valid_metrics.result()
                - A log that contains information about validation
        """
        self.model.eval() # validation mode
        self.valid_metrics.reset()
        with torch.no_grad(): # gradient updates 중단 후 validation 진행
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item()) # validation loss 업데이트
                for met in self.metric_ftns: # validation metric 업데이트
                    self.valid_metrics.update(met.key_name, met.measure(output, target))
        if os.environ['RANK'] == '0':
            self.config.all_logger.wandb_log({'validation_loss':loss}, step=epoch)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """
            * description
                - epoch의 진행상황 출력
            * argument(name : type)
                - batch_idx : int
            * return base.format(current, total, 100.0 * current / total)
                - ex) Train Epoch: 1 [0/5 (0%)] Loss: 1.939521
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size # 현재 진행상황(현재까지 학습된 iteration)
            total = self.data_loader.n_samples # 전체 진행상화(학습해야되는 전체 iteration)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
