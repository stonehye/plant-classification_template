# AI/ML Framework
import torch


class top_k_acc:
    """
        * description
            - top k accuracy 계산
        * argument(name : type)
            - k : int
                - score가 높은 상위 항목들 개수
    """
    def __init__(self, k):
        self.k = k
    
    def measure(self, output, target):
        with torch.no_grad():
            pred = torch.topk(output, self.k, dim=1)[1] # k개 추출
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item() # prediction==label인 데이터 개수를 모두 더함         
        return correct / len(target)

    @property
    def key_name(self):
        key_name = f'{type(self).__name__}'
        for k, v in self.__dict__.items():
            key_name += f'-{k}={v}'
        return key_name