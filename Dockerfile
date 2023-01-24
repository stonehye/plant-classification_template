FROM huggingface/accelerate-gpu:0.9.0
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
#python --version python3.7.11

# 작업디렉토리(default)설정
WORKDIR /home/workspace/
SHELL ["/bin/bash", "-cu"]


# 필수 요소들 설치
RUN \
    apt update -y && apt install -y\
    git\
    axel\
    zip\
    htop\
    screen\
    unzip\
    libgl1-mesa-glx &&\
    apt-get install -y libglib2.0-0 psmisc


COPY . /home/workspace/treebeard_vit
RUN pip install -r /home/workspace/treebeard_vit/requirements.txt

# entrypoint.sh 권한 설정
RUN ["chmod", "+x", "/home/workspace/treebeard_vit/entrypoint.sh"]

# entrypoint.sh 실행
ENTRYPOINT ["bash", "/home/workspace/treebeard_vit/entrypoint.sh"]
