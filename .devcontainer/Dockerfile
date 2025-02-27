FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==22.0
RUN pip3 install PyYAML
RUN pip3 install easydict
RUN pip3 install cython

RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip3 install opencv-python

RUN python3 -m pip install supervisely==6.73.286

RUN pip3 install pandas==2.1.4
RUN pip3 install pycocotools
RUN pip3 install jpeg4py
RUN pip3 install tikzplotlib
RUN pip3 install lmdb
RUN pip3 install scipy
RUN pip3 install visdom
RUN pip3 install timm
RUN pip3 install yacs
RUN pip3 install scikit-image
RUN pip3 install thop
RUN pip3 install tb-nightly

RUN apt-get -y install curl

RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
RUN apt-get install nano

RUN pip3 install gdown

LABEL "role"="development"