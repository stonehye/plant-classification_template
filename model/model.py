# pip module
import numpy as np
# AI/ML Framework
import timm

"""
    model/network에서 학습할 모델을 정의
    모델별로 파일을 따로 생성(ex. renet.py, mnist.py)
    model.py 에서는 model/network에 있는 모델을 호출
"""

def vit_small_patch16_224(num_classes, pretrained=False):
    """
        * description
            - ViTsmall(patch16, 224) Classification Model
        * argument(name : type)
            - num_classes : int
                - classification 모델에서 분류해야될 클래스 개수 지정
            - pretrained : bool
                - 불러올 모델 아키텍처의 사전 학습 여부 설정
    """
    return timm.create_model(
        'vit_small_patch32_224', 
        pretrained=pretrained, num_classes=num_classes
    )
