# built-in
from abc import abstractmethod
# pip module
import numpy as np
# AI/ML Framework
import torch.nn as nn


class BaseModel(nn.Module):
    """
        * description
            - 모든 모델에 대한 Base class
        * inheritance
            - torch.nn.Module
    """
    @abstractmethod
    def forward(self, *inputs):
        """
            * description
                - abstract method
                - Forward pass logic
            * argument(name : type)
                - *inputs : Tuple type
        """
        raise NotImplementedError

    def __str__(self):
        """
            * description
                - 학습가능한 parameter 수 출력
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
