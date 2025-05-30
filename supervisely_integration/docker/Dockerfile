FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==22.0
RUN pip3 install PyYAML==6.0.2
RUN pip3 install easydict==1.13
RUN pip3 install cython==3.0.12

RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip3 install opencv-python

RUN python3 -m pip install supervisely==6.73.369

RUN pip3 install pandas==2.1.4
RUN pip3 install pycocotools==2.0.8
RUN pip3 install jpeg4py==0.1.4
RUN pip3 install tikzplotlib==0.10.1
RUN pip3 install lmdb==1.6.2
RUN pip3 install scipy==1.15.2
RUN pip3 install visdom==0.2.4
RUN pip3 install timm==1.0.14
RUN pip3 install yacs==0.1.8
RUN pip3 install scikit-image==0.25.2
RUN pip3 install thop==0.1.1.post2209072238
RUN pip3 install tb-nightly==2.20.0a20250219

RUN apt-get -y install wget

RUN mkdir -p /weights
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_tiny_1600e_1k.pt -O /weights/fast_itpn_tiny_1600e_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_small_1600e_1k.pt -O /weights/fast_itpn_small_1600e_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_base_clipl_e1600.pt -O /weights/fast_itpn_base_clipl_e1600.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_large_1600e_1k.pt -O /weights/fast_itpn_large_e1600_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_tiny.pth.tar -O /weights/MCITRACK_tiny.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_small.pth.tar -O /weights/MCITRACK_small.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_base.pth.tar -O /weights/MCITRACK_base.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_large.pth.tar -O /weights/MCITRACK_large.pth.tar

LABEL python_sdk_version=6.73.369