# pip module
import wandb
from accelerate.logging import get_logger
from accelerate import logging
import logging.config


class Logger:
    """
        * description
            - training logger
            - Wandb, Accelerator 포함
        * argument(name : type)
            - save_dir : str
            - log_config : dict
    """
    def __init__(self, save_dir, local_rank, log_config:dict):
        logging.config.dictConfig(log_config['log_config']) # log config

        # wandb options
        if local_rank == '0':
            wandb.login(key=log_config['wandb']['WANDB_API_KEY'])
            wandb.init(project = log_config['wandb']['project'], entity = log_config['wandb']['entity'], config=log_config['wandb']['config'], dir=log_config['wandb']['dir'])     
            wandb.run.name = log_config['wandb']['run_name'] # wandb run 이름 설정
            self.log_freq = log_config['wandb']['log_freq'] # log frequency 설정
        
        # accelerate logger options
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        
    def wandb_log(self, dic, step=None):
        # wandb log
        if step==None:
            wandb.log(dic)
        else:
            wandb.log(dic, step)

    def wandb_watch(self, model, criterion, log_freq=None):
        # wandb watch
        _log_freq = log_freq if log_freq else self.log_freq
        wandb.watch(model, criterion, log_freq=_log_freq)
        
    def wandb_best_alert(self, epoch, acc):
        wandb.alert(
            title='Best accuracy alert', 
            text=f'Epoch: {epoch}, Accuracy: {acc}'
        )

    def get_logger(self, name, verbosity=2):
        # get logger(accelerator)
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = get_logger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
    